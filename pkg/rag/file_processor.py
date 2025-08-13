# /workspace/LiveTalking/pkg/rag/file_processor.py
# -*- coding: utf-8 -*-

"""
中文说明（稳定方案）：
- PDF 默认使用 --redo-ocr（对已有可见文本的页面，仅对图片区域做 OCR），去掉 --deskew（与 redo-ocr 不兼容）
- 若 redo-ocr 失败，自动回退到 --force-ocr + --deskew（整页栅格化，体积变大，但兜底稳）
- 输出 PDF 丢弃到 /dev/null，仅保留 sidecar 文本；仅捕获 stderr 作为日志，避免 stdout 二进制被误解码
- aiohttp 的 HTTP 错误 reason 统一清洗为单行，避免 “Reason cannot contain \\n”
"""

import asyncio               # 异步：把CPU密集型任务丢到线程池，避免阻塞事件循环
import locale                # 日志解码：按系统首选编码解码stderr
import os                    # 使用 os.devnull 丢弃PDF输出；环境变量可扩展策略
import re                    # 清洗HTTP错误reason，去掉换行
import subprocess            # 子进程运行 ocrmypdf
import time                  # 原子写入：定义临时 sidecar 文件路径
from pathlib import Path

import ocrmypdf              # 存在性/版本校验（不直接走其Python API）
from aiohttp import web

from logger import logger
from pkg.rag.config import ALLOWED_EXTENSIONS, KB_ROOT_PATH, OCR_LANGUAGE
from pkg.rag.file_converter import convert_to_pdf


class FileProcessor:
    """
    文件处理类：保存上传文件 -> 处理为 TXT（文本/转换后PDF/经OCR）
    """

    @staticmethod
    async def save_and_process_file(file_bytes: bytes, original_filename: str, kb_name: str) -> Path:
        """
        异步入口：保存上传文件并调用同步处理；返回最终 TXT 路径
        - 修复：HTTP 错误 reason 必须单行
        """
        file_ext = Path(original_filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise web.HTTPBadRequest(reason=f"不支持的文件类型: {file_ext}。只支持 {', '.join(ALLOWED_EXTENSIONS)}")

        kb_path = KB_ROOT_PATH / kb_name
        kb_path.mkdir(exist_ok=True)
        save_path = kb_path / original_filename

        try:
            with open(save_path, 'wb') as f:
                f.write(file_bytes)
            logger.info(f"文件 '{original_filename}' 已成功保存到 '{save_path}'")
        except IOError as e:
            logger.error(f"保存文件时出错 '{save_path}': {e}")
            raise web.HTTPInternalServerError(reason="保存上传文件失败。")

        loop = asyncio.get_event_loop()
        try:
            txt_path = await loop.run_in_executor(
                None,
                FileProcessor._process_file_sync,
                save_path,
                kb_path,
                file_ext
            )
            return txt_path
        except Exception as e:
            logger.error(f"处理文件 '{original_filename}' 时发生严重错误: {e}", exc_info=True)
            # ✅ aiohttp 的 reason 不允许换行，做单行清洗
            single_line_reason = re.sub(r"[\r\n]+", " ", f"{e}")
            # 如需保留原始文件用于排查，可注释掉以下删除
            if save_path.exists():
                save_path.unlink()
            raise web.HTTPInternalServerError(reason=f"处理文件 '{original_filename}' 失败: {single_line_reason}")

    @staticmethod
    def _process_file_sync(file_path: Path, kb_path: Path, file_ext: str) -> Path:
        """
        同步处理主流程（在线程池中执行）：
        - PDF：调用升级版OCR流程
        - Office：先转 PDF 再 OCR
        - 纯文本：直接读写
        """
        txt_path = kb_path / f"{file_path.stem}.txt"

        office_formats = [".docx", ".doc", ".pptx", ".ppt"]
        text_formats = [".txt", ".md"]

        if file_ext == ".pdf":
            FileProcessor.process_pdf_upgraded(file_path, txt_path)
            # 处理完后删除原始PDF文件
            if file_path.exists():
                file_path.unlink()
        elif file_ext in office_formats:
            pdf_path = None
            try:
                pdf_path = convert_to_pdf(file_path, kb_path)
                FileProcessor.process_pdf_upgraded(pdf_path, txt_path)
            finally:
                # 清理原始office文件和转换过程中的PDF
                if file_path.exists():
                    file_path.unlink()
                if pdf_path and pdf_path.exists():
                    pdf_path.unlink()
        elif file_ext in text_formats:
            # [!!! 最终修复 !!!]
            # 对于纯文本文件，原始路径(file_path)和目标路径(txt_path)是同一个文件。
            # 这里只需确保它存在即可，无需额外处理。删除原始文件的逻辑会导致文件被错误地删除。
            FileProcessor.process_text(file_path, txt_path)
            # 如果原始文件名不是.txt结尾（例如.md），则在处理后删除原始文件
            if file_path.suffix != ".txt":
                if file_path.exists():
                    file_path.unlink()


        return txt_path

    @staticmethod
    def process_text(text_file_path: Path, txt_path: Path):
        """
        读取文本文件为 UTF-8（宽松替换非法字节），写入目标路径
        """
        logger.info(f"正在处理文本文件: {text_file_path}...")
        try:
            # 如果源文件和目标文件是同一个，就不需要做任何事
            if text_file_path == txt_path:
                logger.info(f"源文件和目标文件路径相同，无需处理: {txt_path}")
                return

            # 如果路径不同（例如，从.md转.txt），则执行复制/重命名
            with open(text_file_path, 'rb') as f_in:
                raw_content = f_in.read()
            
            content = raw_content.decode('utf-8', errors='replace')
            
            with open(txt_path, 'w', encoding='utf-8') as f_out:
                f_out.write(content)

            logger.info(f"成功将文本内容保存到: {txt_path}")
        except Exception as e:
            logger.error(f"处理文本文件 {text_file_path} 时出错: {e}")
            raise

    @staticmethod
    def process_pdf_upgraded(pdf_path: Path, txt_path: Path):
        """
        使用两阶段 OCR 策略和原子写入，对PDF文件进行健壮的处理。
        """
        logger.info(f"正在使用升级版 OCR 参数处理 PDF: {pdf_path}...")
        
        tmp_txt_path = txt_path.with_suffix(".txt.tmp")
        
        # 策略一：尝试 --redo-ocr
        cmd_redo = [
            "ocrmypdf", "-l", OCR_LANGUAGE, "--redo-ocr",
            "--rotate-pages", "--optimize", "1", "--oversample", "300",
            "--tesseract-timeout", "300", "--skip-big", "300",
            "--sidecar", str(tmp_txt_path), str(pdf_path), "-",
        ]

        # 策略二：回退方案
        cmd_force = [
            "ocrmypdf", "-l", OCR_LANGUAGE, "--force-ocr",
            "--rotate-pages", "--deskew", "--optimize", "1",
            "--oversample", "300", "--tesseract-timeout", "300", "--skip-big", "300",
            "--sidecar", str(tmp_txt_path), str(pdf_path), "-",
        ]
        
        success = False
        last_error = ""
        encoding = locale.getpreferredencoding(False)

        # 首先尝试策略一
        try:
            logger.info("OCR 尝试 #1: 使用 --redo-ocr 策略...")
            proc = subprocess.run(cmd_redo, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            if proc.stderr:
                logger.info(f"OCRmyPDF (--redo-ocr) 日志:\n{proc.stderr.decode(encoding, errors='ignore')}")
            success = True
        except subprocess.CalledProcessError as e:
            last_error = e.stderr.decode(encoding, errors='ignore')
            logger.warning(f"--redo-ocr 策略失败，将尝试 --force-ocr。错误: {last_error}")
        except FileNotFoundError:
            logger.error("命令 'ocrmypdf' 未找到。")
            raise RuntimeError("OCR 工具 (ocrmypdf) 不可用。")

        # 检查是否需要兜底
        need_fallback = False
        if not success:
            need_fallback = True
        elif tmp_txt_path.exists():
            try:
                if tmp_txt_path.stat().st_size < 200: # 检查文件大小
                     logger.warning(f"策略一提取的文本量过少，将尝试兜底策略")
                     need_fallback = True
            except Exception as e:
                logger.warning(f"读取临时文本文件大小失败: {e}")
                need_fallback = True
        
        # 如果需要，尝试策略二
        if need_fallback:
            try:
                logger.info("OCR 尝试 #2: 使用 --force-ocr + --deskew 策略...")
                proc = subprocess.run(cmd_force, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
                if proc.stderr:
                    logger.info(f"OCRmyPDF (--force-ocr) 日志:\n{proc.stderr.decode(encoding, errors='ignore')}")
                success = True
            except subprocess.CalledProcessError as e:
                last_error = e.stderr.decode(encoding, errors='ignore')
                logger.error(f"--force-ocr 策略也失败了。错误: {last_error}")
                success = False # 明确设置失败
            except FileNotFoundError:
                 logger.error("命令 'ocrmypdf' 未找到。")
                 raise RuntimeError("OCR 工具 (ocrmypdf) 不可用。")

        if not success:
            clean_error = last_error.replace('\\n', ' ').replace('\\r', '')
            raise RuntimeError(f"所有 OCR 策略均失败: {clean_error}")

        if tmp_txt_path.exists():
            os.replace(tmp_txt_path, txt_path)
            logger.info(f"成功将 PDF-OCR 文本原子化写入到: {txt_path}")
        else:
            txt_path.touch()
            logger.warning(f"OCR 过程没有生成文本文件，可能PDF为空。已创建空的标记文件: {txt_path}")

    @staticmethod
    async def cleanup_kb_files(kb_name: str):
        """
        删除与指定知识库相关的所有文件
        """
        kb_path = KB_ROOT_PATH / kb_name
        if not kb_path.is_dir():
            return

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, FileProcessor._delete_directory, kb_path)
            logger.info(f"已成功删除知识库目录: {kb_path}")
        except Exception as e:
            logger.error(f"删除知识库目录 '{kb_path}' 时出错: {e}")
            raise

    @staticmethod
    def _delete_directory(path: Path):
        import shutil
        shutil.rmtree(path)