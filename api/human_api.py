# /workspace/LiveTalking/api/human_api.py
# -*- 简化注释：导入必要的库 -*-
import asyncio
import json

from aiohttp import web

from api.utils import transcribe_audio
from logger import logger

async def _handle_llm_conversation(llm_client, text, session_id, nerfreals, tts_options):
    """
    处理标准的LLM对话流程。
    """
    async def stream_llm_to_tts():
        try:
            response_chunks = []
            async for text_chunk in llm_client.ask(text):
                if session_id not in nerfreals:
                    logger.warning(f"会话 {session_id} 在接收LLM响应时已关闭，中断任务。")
                    return
                response_chunks.append(text_chunk)
            
            full_response = "".join(response_chunks)
            if full_response and session_id in nerfreals:
                nerfreals[session_id].put_msg_txt(full_response, tts_options=tts_options)
        except Exception as e:
            logger.error(f"标准LLM对话任务执行出错 (会话 {session_id}): {e}")

    asyncio.create_task(stream_llm_to_tts())


async def _handle_rag_conversation(rag_core, text, session_id, nerfreals, tts_options):
    """
    处理RAG增强的对话流程。
    """
    async def stream_rag_to_tts():
        try:
            response_chunks = []
            # RAG核心现在使用共享的LLM客户端，所以我们直接从它那里获取响应
            async for text_chunk in rag_core.get_response(text):
                if session_id not in nerfreals:
                    logger.warning(f"会话 {session_id} 在接收RAG响应时已关闭，中断任务。")
                    return
                response_chunks.append(text_chunk)

            full_response = "".join(response_chunks)
            if full_response and session_id in nerfreals:
                nerfreals[session_id].put_msg_txt(full_response, tts_options=tts_options)
        except Exception as e:
            logger.error(f"RAG对话任务执行出错 (会话 {session_id}): {e}")

    asyncio.create_task(stream_rag_to_tts())


async def human(request):
    """
    处理文本输入，根据参数选择标准对话或RAG增强对话。
    """
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        
        nerfreals = request.app['nerfreals']
        llm_client = request.app['llm_client']
        rag_core = request.app['rag_core']

        if not sessionid or sessionid not in nerfreals:
            raise ValueError("无效或已过期的 sessionid")

        if params.get('interrupt'):
            nerfreals[sessionid].flush_talk()

        tts_options = params.get('tts_options', {})
        use_rag = params.get('use_rag', False)
        kb_name = params.get('kb_name')

        # 根据use_rag标志，决定对话流程
        if use_rag:
            # 在RAG模式下，首先确保设置了当前知识库
            rag_core.set_current_kb(kb_name)
            await _handle_rag_conversation(rag_core, params['text'], sessionid, nerfreals, tts_options)
        else:
            # 标准LLM对话
            await _handle_llm_conversation(llm_client, params['text'], sessionid, nerfreals, tts_options)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "ok"})
        )
    except Exception as e:
        logger.exception('human接口异常:')
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"code": -1, "msg": str(e)})
        )


async def humanaudio(request):
    """
    处理上传的音频文件，直接驱动虚拟人。
    """
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        nerfreals = request.app['nerfreals']
        
        if sessionid in nerfreals:
            fileobj = form["file"]
            filebytes = fileobj.file.read()
            nerfreals[sessionid].put_audio_file(filebytes)
            
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
    except Exception as e:
        logger.exception('humanaudio接口异常:')
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"code": -1, "msg": str(e)})
        )


async def audio_chat(request):
    """
    处理语音聊天，根据参数选择标准对话或RAG增强对话。
    """
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        
        nerfreals = request.app['nerfreals']
        asr_model = request.app['asr_model']
        llm_client = request.app['llm_client']
        rag_core = request.app['rag_core']

        if not sessionid or sessionid not in nerfreals:
            return web.Response(
                content_type="application/json", status=400,
                text=json.dumps({"code": -1, "msg": "需要有效的sessionid"})
            )

        fileobj = form.get("file")
        if not fileobj:
            return web.Response(
                content_type="application/json", status=400,
                text=json.dumps({"code": -1, "msg": "需要音频文件"})
            )

        audio_bytes = fileobj.file.read()
        transcribed_text = await transcribe_audio(asr_model, audio_bytes)
        
        if not transcribed_text:
            logger.info("ASR未能识别出文本，不作处理。")
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ASR未能识别出文本。"})
            )
        
        logger.info(f"ASR识别结果: {transcribed_text}")

        # 从表单中获取RAG相关参数和TTS选项
        use_rag = form.get('use_rag', 'false').lower() == 'true'
        kb_name = form.get('kb_name')
        tts_options_str = form.get('tts_options', '{}')
        tts_options = json.loads(tts_options_str)

        if use_rag:
            rag_core.set_current_kb(kb_name)
            await _handle_rag_conversation(rag_core, transcribed_text, sessionid, nerfreals, tts_options)
        else:
            await _handle_llm_conversation(llm_client, transcribed_text, sessionid, nerfreals, tts_options)

        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
    except Exception as e:
        logger.exception('audio_chat接口异常:')
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"code": -1, "msg": str(e)})
        )


def register_human_routes(app):
    """
    将与虚拟人交互相关的路由注册到 aiohttp 应用。
    """
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/audio_chat", audio_chat)
