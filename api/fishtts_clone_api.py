# /workspace/LiveTalking/api/fishtts_clone_api.py
# -*- coding: utf-8 -*-
# 目标：实现一个带自动格式转换功能的、鲁棒的FishTTS克隆音色管理API。
# 说明：完整采纳您提供的优秀代码，实现了格式转换、异步处理和RESTful接口。

import os
import re
import shutil
from pathlib import Path
import soundfile as sf
from aiohttp import web
from logger import logger
import io
import asyncio      # 新增: 用于异步执行命令行
import tempfile     # 新增: 用于创建临时文件

# --- 辅助函数 ---

def get_voice_list(path: Path) -> list:
    """获取声音列表，不含文件扩展名"""
    if not path.exists():
        return []
    # 确保只返回 .wav 文件的文件名（不带后缀）
    return sorted([p.stem for p in path.glob('*.wav')])

async def validate_audio(audio_bytes: bytes) -> bool:
    """验证音频时长是否在40秒内"""
    try:
        with io.BytesIO(audio_bytes) as f:
            # 使用 soundfile 读取音频数据
            data, samplerate = sf.read(f)
            # 计算时长
            duration = len(data) / samplerate
            if duration > 40:
                logger.warning(f"音频验证失败：时长 {duration:.2f}s > 40s")
                return False
            return True
    except Exception as e:
        logger.error(f"解析音频文件时出错: {e}")
        return False

# --- API 处理器 ---

async def list_cloned_voices(request: web.Request):
    """获取所有已克隆的FishTTS声音列表"""
    try:
        # 假设 opt.fishtts_cloned_voices_path 在 app.py 中定义并传入
        opt = request.app['opt']
        clone_path = Path(opt.fishtts_cloned_voices_path)
        voices = get_voice_list(clone_path)
        return web.json_response({"voices": voices})
    except Exception as e:
        logger.exception('list_cloned_voices 接口异常:')
        return web.json_response({"error": str(e)}, status=500)

async def upload_clone_voice(request: web.Request):
    """上传新的克隆声音源，并对非wav格式进行转换"""
    try:
        opt = request.app['opt']
        clone_path = Path(opt.fishtts_cloned_voices_path)
        # 确保目录存在
        clone_path.mkdir(parents=True, exist_ok=True)

        # 1. 检查仓库容量
        if len(get_voice_list(clone_path)) >= 10:
            return web.json_response({"error": "克隆声音仓库已满 (最多10个)"}, status=400)

        data = await request.post()

        # 2. 验证输入字段
        voice_name = data.get('voice_name')
        audio_file_field = data.get('audio_file')

        if not voice_name or not audio_file_field:
            return web.json_response({"error": "缺少 voice_name 或 audio_file 字段"}, status=400)

        # 3. 验证声音名称格式
        if not re.match(r'^[a-zA-Z0-9_]{3,}$', voice_name):
            return web.json_response({"error": "声音名称不合法 (必须是3位以上的字母、数字或下划线)"}, status=400)

        if (clone_path / f"{voice_name}.wav").exists():
            return web.json_response({"error": f"声音名称 '{voice_name}' 已存在"}, status=409) # 409 Conflict 更合适

        # 4. 验证文件类型
        file_name = audio_file_field.filename
        if not file_name.lower().endswith(('.wav', '.mp3', '.flac')):
            return web.json_response({"error": "不支持的文件类型，请上传 .wav, .mp3, 或 .flac 格式"}, status=400)

        audio_bytes = audio_file_field.file.read()
        
        # 5. 【核心】格式转换 (如果不是 wav)
        if not file_name.lower().endswith('.wav'):
            logger.info(f"检测到非wav格式 ({Path(file_name).suffix}), 开始使用 ffmpeg 转换...")
            # 创建一个带正确后缀的临时文件来保存上传的音频
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp_in:
                tmp_in.write(audio_bytes)
                tmp_in_path = tmp_in.name
            
            # 定义输出的 wav 临时文件路径
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
                tmp_out_path = tmp_out.name

            try:
                # 使用 ffmpeg 进行转换, 统一转换为 16kHz, 16-bit, 单声道 WAV
                command = [
                    'ffmpeg', '-i', tmp_in_path, 
                    '-ar', '16000',           # 设置采样率为 16000 Hz
                    '-ac', '1',               # 设置为单声道
                    '-c:a', 'pcm_s16le',      # 设置编码为 16-bit PCM
                    '-y', tmp_out_path        # 如果输出文件已存在则覆盖
                ]
                # 异步执行 ffmpeg 命令
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()

                # 检查转换是否成功
                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    logger.error(f"FFmpeg 转换失败: {error_msg}")
                    return web.json_response({"error": f"音频转换失败: {error_msg}"}, status=500)
                
                logger.info("FFmpeg 转换成功.")
                # 读取转换后的 wav 文件内容
                with open(tmp_out_path, 'rb') as f_out:
                    audio_bytes = f_out.read()
            
            finally:
                # 清理临时文件，无论成功与否
                if os.path.exists(tmp_in_path):
                    os.remove(tmp_in_path)
                if os.path.exists(tmp_out_path):
                    os.remove(tmp_out_path)

        # 6. 验证音频时长 (对原始或转换后的wav文件进行验证)
        if not await validate_audio(audio_bytes):
            return web.json_response({"error": "音频时长超过40秒限制"}, status=400)
            
        # 7. 保存最终的 wav 文件
        save_path = clone_path / f"{voice_name}.wav"
        
        try:
            # 直接将最终的 wav 格式的 bytes 写入文件
            with open(save_path, 'wb') as f:
                f.write(audio_bytes)
        except Exception as e:
            logger.error(f"保存转换后的音频文件时出错: {e}")
            return web.json_response({"error": f"保存音频文件失败: {e}"}, status=500)

        logger.info(f"成功添加新的克隆声音: {voice_name}")
        
        voices = get_voice_list(clone_path)
        return web.json_response({
            "status": "success",
            "voice_name": voice_name,
            "voices": voices
        })

    except Exception as e:
        logger.exception('upload_clone_voice 接口异常:')
        return web.json_response({"error": str(e)}, status=500)


async def delete_clone_voice(request: web.Request):
    """删除指定的克隆声音"""
    try:
        voice_name = request.match_info.get('voice_name')
        if not voice_name:
            return web.json_response({"error": "未提供声音名称"}, status=400)

        opt = request.app['opt']
        clone_path = Path(opt.fishtts_cloned_voices_path)
        
        file_path = clone_path / f"{voice_name}.wav"

        if not file_path.exists():
            return web.json_response({"error": f"声音 '{voice_name}' 不存在"}, status=404)
        
        os.remove(file_path)
        logger.info(f"成功删除克隆声音: {voice_name}")

        voices = get_voice_list(clone_path)
        return web.json_response({
            "status": "success",
            "deleted_voice": voice_name,
            "voices": voices
        })

    except Exception as e:
        logger.exception('delete_clone_voice 接口异常:')
        return web.json_response({"error": str(e)}, status=500)

# --- 路由注册 ---

def register_fishtts_clone_routes(app: web.Application):
    """将FishTTS克隆相关的路由注册到 aiohttp 应用"""
    app.router.add_get("/fishtts/voices", list_cloned_voices)
    app.router.add_post("/fishtts/voices", upload_clone_voice)
    app.router.add_delete("/fishtts/voices/{voice_name}", delete_clone_voice)