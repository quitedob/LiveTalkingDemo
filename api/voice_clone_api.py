# /workspace/LiveTalking/api/voice_clone_api.py
# -*- 简化注释：导入必要的库 -*-
import asyncio
import json

from aiohttp import web

from logger import logger


async def get_cloned_voices(request):
    """
    获取所有已克隆的声音配置文件列表。
    Retrieves a list of all cloned voice profiles.
    """
    # 从应用上下文中获取CosyVoice实例
    cosyvoice_tts_instance = request.app.get('cosyvoice_tts_instance')
    if not cosyvoice_tts_instance:
        return web.Response(
            content_type="application/json", status=503,
            text=json.dumps({"error": "CosyVoice service not available"})
        )
    try:
        # 从声音配置文件管理器加载配置
        voices = cosyvoice_tts_instance.voice_profile_manager.load_profiles()
        return web.Response(
            content_type="application/json",
            text=json.dumps({"voices": voices})
        )
    except Exception as e:
        logger.exception('get_cloned_voices 接口异常:')
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"error": str(e)})
        )


async def clone_voice(request):
    """
    处理音频上传，克隆新的声音。
    Handles audio upload to clone a new voice.
    """
    cosyvoice_tts_instance = request.app.get('cosyvoice_tts_instance')
    if not cosyvoice_tts_instance:
        return web.Response(
            content_type="application/json", status=503,
            text=json.dumps({"error": "CosyVoice service not available"})
        )
    try:
        data = await request.post()
        voice_name = data.get('voice_name')
        audio_file_field = data.get('audio_file')

        if not voice_name or not audio_file_field:
            return web.Response(
                content_type="application/json", status=400,
                text=json.dumps({"error": "Missing voice_name or audio_file"})
            )

        audio_bytes = audio_file_field.file.read()

        # 在线程池中执行耗时的声音克隆操作
        new_voice_profile = await asyncio.get_event_loop().run_in_executor(
            None,
            cosyvoice_tts_instance.add_cloned_voice,
            voice_name,
            audio_bytes
        )
        # 重新加载所有声音配置
        all_voices = cosyvoice_tts_instance.voice_profile_manager.load_profiles()
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "status": "success",
                "new_voice": new_voice_profile,
                "voices": all_voices
            })
        )
    except Exception as e:
        logger.exception('clone_voice 接口异常:')
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"error": str(e)})
        )


def register_voice_clone_routes(app):
    """
    将声音克隆相关的路由注册到 aiohttp 应用。
    Registers voice cloning related routes to the aiohttp application.
    """

    app.router.add_get("/get_cloned_voices", get_cloned_voices)
    app.router.add_post("/clone_voice", clone_voice)

