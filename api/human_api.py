# /workspace/LiveTalking/api/human_api.py
# -*- 简化注释：导入必要的库 -*-
import asyncio
import json
import re

from aiohttp import web

from api.utils import transcribe_audio
from logger import logger

async def _llm_to_sentence_stream(llm_stream):
    """
    将LLM的字符流转换为句子流。
    Converts a stream of characters from LLM into a stream of sentences.
    """
    buffer = ""
    # 使用正则表达式按常见的结束标点分割句子
    sentence_delimiters = re.compile(r'([。？！；!?;\n])')
    
    async for chunk in llm_stream:
        buffer += chunk
        while True:
            match = sentence_delimiters.search(buffer)
            if match:
                sentence = buffer[:match.end()]
                buffer = buffer[match.end():]
                yield sentence.strip()
            else:
                break
    # 处理流结束后剩余的文本
    if buffer.strip():
        yield buffer.strip()


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

        tts_options = params.get('tts_options', {})
        use_rag = params.get('use_rag', False)
        kb_name = params.get('kb_name')

        async def stream_pipeline(text):
            try:
                if use_rag:
                    rag_core.set_current_kb(kb_name)
                    llm_stream = rag_core.get_response(text)
                else:
                    llm_stream = llm_client.ask(text)
                
                sentence_stream = _llm_to_sentence_stream(llm_stream)
                
                if sessionid in nerfreals:
                    # ======================================================
                    # ★★★ 核心修复：使用 **tts_options 解包字典 ★★★
                    # ======================================================
                    # 之前的代码:
                    # await nerfreals[sessionid].put_msg_txt(sentence_stream, tts_options=tts_options)
                    # 修复后的代码:
                    await nerfreals[sessionid].put_msg_txt(sentence_stream, **tts_options)
                    # ======================================================
                else:
                    logger.warning(f"会话 {sessionid} 在处理流式响应时已关闭。")

            except Exception as e:
                logger.error(f"流式处理管道出错 (会话 {sessionid}): {e}", exc_info=True)

        # 创建一个后台任务来处理整个流式对话，立即返回HTTP响应
        asyncio.create_task(stream_pipeline(params['text']))

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

        async def stream_pipeline(text):
            try:
                if use_rag:
                    rag_core.set_current_kb(kb_name)
                    llm_stream = rag_core.get_response(text)
                else:
                    llm_stream = llm_client.ask(text)
                
                sentence_stream = _llm_to_sentence_stream(llm_stream)
                
                if sessionid in nerfreals:
                    # ======================================================
                    # ★★★ 核心修复：使用 **tts_options 解包字典 ★★★
                    # ======================================================
                    await nerfreals[sessionid].put_msg_txt(sentence_stream, **tts_options)
                    # ======================================================
                else:
                    logger.warning(f"会话 {sessionid} 在处理语音聊天的流式响应时已关闭。")

            except Exception as e:
                logger.error(f"语音聊天的流式处理管道出错 (会话 {sessionid}): {e}", exc_info=True)

        asyncio.create_task(stream_pipeline(transcribed_text))

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