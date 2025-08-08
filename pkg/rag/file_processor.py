# /workspace/LiveTalking/pkg/rag/file_processor.py
import asyncio
import locale
import subprocess
from pathlib import Path

import ocrmypdf
from docx import Document
from aiohttp import web

from logger import logger
from pkg.rag.config import ALLOWED_EXTENSIONS, KB_ROOT_PATH, OCR_LANGUAGE


class FileProcessor:
    """
    文件处理类，负责处理上传的文件，包括格式转换和文本提取。
    """

    @staticmethod
    async def save_and_process_file(file_bytes: bytes, original_filename: str, kb_name: str) -> Path:
        """
        保存上传的文件并根据其类型进行处理，最终返回提取出的文本文件路径。
        :param file_bytes: 文件的字节内容。
        :param original_filename: 原始文件名。
        :param kb_name: 知识库名称。
        :return: 提取出的文本文件路径。
        """
        # 1. 验证文件扩展名
        file_ext = Path(original_filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise web.HTTPBadRequest(reason=f"不支持的文件类型: {file_ext}。只支持 {', '.join(ALLOWED_EXTENSIONS)}")

        # 2. 创建知识库子目录并保存原始文件
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
            
        # 3. 根据文件类型进行处理
        txt_path = kb_path / f"{Path(original_filename).stem}.txt"
        
        # 在事件循环的线程池中运行可能阻塞的IO或子进程操作
        loop = asyncio.get_event_loop()

        try:
            if file_ext == ".pdf":
                await loop.run_in_executor(None, FileProcessor.process_pdf, save_path, txt_path)
            elif file_ext in [".docx", ".doc"]:
                # 对于.doc，我们尝试用同样的方式处理，因为python-docx有时能处理.doc
                await loop.run_in_executor(None, FileProcessor.process_docx, save_path, txt_path)
            elif file_ext in [".txt", ".md"]:
                # 对于纯文本或Markdown，直接重命名（或复制内容）
                save_path.rename(txt_path)
                logger.info(f"已将 '{save_path}' 重命名为 '{txt_path}'")
            
            return txt_path
        except Exception as e:
            logger.error(f"处理文件 '{original_filename}' 时发生严重错误: {e}")
            raise web.HTTPInternalServerError(reason=f"处理文件 '{original_filename}' 失败。")

    @staticmethod
    def process_pdf(pdf_path: Path, txt_path: Path):
        """
        使用ocrmypdf对PDF文件进行OCR处理，并将结果保存到文本文件。
        """
        logger.info(f"正在使用 OCR 处理 PDF 文件: {pdf_path}...")
        try:
            # 使用 ocrmypdf API 进行处理
            ocrmypdf.ocr(
                pdf_path,
                pdf_path, # 直接覆盖原文件（OCR后）
                output_type='pdf',
                sidecar=txt_path,
                language=OCR_LANGUAGE,
                force_ocr=True, # 强制对所有页面进行OCR
                clean=True, # 清理页面
                deskew=True, # 自动校正倾斜
            )
            logger.info(f"成功将 PDF-OCR 文本提取到: {txt_path}")
        except ocrmypdf.exceptions.EncryptedPdfError:
            logger.error(f"错误: PDF 文件 '{pdf_path}' 已加密，无法处理。")
            raise Exception("加密的PDF文件")
        except ocrmypdf.exceptions.PriorOcrFoundError:
             # 如果已经有OCR层，我们可以选择直接提取文本或强制重新OCR
            logger.warning(f"PDF '{pdf_path}' 中检测到已有文本层。将直接提取文本。")
            # 这里可以添加一个纯文本提取的逻辑作为备选
            # 为了简化，我们仍然让 sidecar 生效
            pass
        except Exception as e:
            logger.error(f"使用 ocrmypdf 处理 PDF 时出错: {e}")
            raise

    @staticmethod
    def process_docx(docx_path: Path, txt_path: Path):
        """
        从DOCX文件中提取文本并保存到文本文件。
        """
        logger.info(f"正在从 DOCX 文件提取文本: {docx_path}...")
        try:
            document = Document(docx_path)
            full_text = [para.text for para in document.paragraphs]
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(full_text))
            logger.info(f"成功将 DOCX 文本提取到: {txt_path}")
        except Exception as e:
            logger.error(f"处理 DOCX 文件时出错: {e}")
            raise

    @staticmethod
    async def cleanup_kb_files(kb_name: str):
        """
        删除与指定知识库相关的所有文件。
        """
        kb_path = KB_ROOT_PATH / kb_name
        if not kb_path.is_dir():
            logger.warning(f"尝试删除不存在的知识库目录: {kb_path}")
            return
        
        loop = asyncio.get_event_loop()
        try:
            # 在线程池中执行删除操作，避免阻塞
            await loop.run_in_executor(None, FileProcessor._delete_directory, kb_path)
            logger.info(f"已成功删除知识库目录: {kb_path}")
        except Exception as e:
            logger.error(f"删除知识库目录 '{kb_path}' 时出错: {e}")
            raise
    
    @staticmethod
    def _delete_directory(path: Path):
        """递归删除目录及其所有内容的辅助函数"""
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                FileProcessor._delete_directory(child)
        path.rmdir()

