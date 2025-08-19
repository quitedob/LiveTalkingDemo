# /workspace/LiveTalking/api/human_api.py
# -*- 简化注释：导入必要的库 -*-
import asyncio
import json
import re
import ollama
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

async def describe_image(request: web.Request):
    """
    POST /vision/describe - 接收图片和可选提示，返回模型的文字描述。
    这是一个不与数字人直接绑定的通用视觉 API。
    """
    try:
        # 使用 multipart/form-data 来接收文件和文本字段
        reader = await request.multipart()
        image_bytes = None
        prompt = "请详细描述这张图片的内容。" # 默认提示词
        model = "gemma3:12b" # 默认模型

        async for field in reader:
            if field.name == 'image':
                image_bytes = await field.read()
            elif field.name == 'prompt':
                prompt = await field.text()
            elif field.name == 'model':
                model = await field.text()

        if not image_bytes:
            return web.json_response({"error": "缺少 'image' 文件字段"}, status=400)

        # 在线程池中执行同步的 Ollama 调用，避免阻塞 aiohttp 的事件循环
        loop = asyncio.get_event_loop()
        description = await loop.run_in_executor(
            None, 
            _describe_image_sync, 
            image_bytes, 
            prompt, 
            model
        )

        return web.json_response({"description": description})

    except Exception as e:
        logger.exception('describe_image 接口异常:')
        return web.json_response({"error": str(e)}, status=500)

async def human(request):
    """
    处理文本输入，根据参数选择标准对话或RAG增强对话。
    返回流式响应，包含LLM响应。
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
        text = params.get('text', '')

        # 创建流式响应
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'application/json'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        await response.prepare(request)

        async def stream_pipeline(input_text):
            try:
                if use_rag:
                    rag_core.set_current_kb(kb_name)
                    llm_stream = rag_core.get_response(input_text)
                else:
                    llm_stream = llm_client.ask(input_text)
                
                # 收集完整的LLM响应用于流式传输
                full_response = ""
                async for chunk in llm_stream:
                    full_response += chunk
                    # 发送流式LLM响应到前端
                    chunk_data = json.dumps({"response": chunk}) + "\n"
                    await response.write(chunk_data.encode('utf-8'))
                
                # 发送完整响应
                final_data = json.dumps({"llm_response": full_response}) + "\n"
                await response.write(final_data.encode('utf-8'))
                
                # 同时驱动数字人
                sentence_stream = _llm_to_sentence_stream(llm_client.ask(input_text) if not use_rag else rag_core.get_response(input_text))
                
                if sessionid in nerfreals:
                    # ======================================================
                    # ★★★ 核心修复：使用 **tts_options 解包字典 ★★★
                    # ======================================================
                    await nerfreals[sessionid].put_msg_txt(sentence_stream, **tts_options)
                    # ======================================================
                else:
                    logger.warning(f"会话 {sessionid} 在处理流式响应时已关闭。")

            except Exception as e:
                logger.error(f"流式处理管道出错 (会话 {sessionid}): {e}", exc_info=True)
                error_data = json.dumps({"error": str(e)}) + "\n"
                await response.write(error_data.encode('utf-8'))
            finally:
                await response.write_eof()

        # 执行流式处理
        await stream_pipeline(text)
        return response
        
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
    返回流式响应，包含ASR结果和LLM响应。
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

        # 创建流式响应
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'application/json'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        await response.prepare(request)

        # 立即发送ASR结果
        asr_data = json.dumps({"asr_result": transcribed_text}) + "\n"
        await response.write(asr_data.encode('utf-8'))

        async def stream_pipeline(text):
            try:
                if use_rag:
                    rag_core.set_current_kb(kb_name)
                    llm_stream = rag_core.get_response(text)
                else:
                    llm_stream = llm_client.ask(text)
                
                # 收集完整的LLM响应用于流式传输
                full_response = ""
                async for chunk in llm_stream:
                    full_response += chunk
                    # 发送流式LLM响应到前端
                    chunk_data = json.dumps({"response": chunk}) + "\n"
                    await response.write(chunk_data.encode('utf-8'))
                
                # 发送完整响应
                final_data = json.dumps({"llm_response": full_response}) + "\n"
                await response.write(final_data.encode('utf-8'))
                
                # 同时驱动数字人
                sentence_stream = _llm_to_sentence_stream(llm_client.ask(text) if not use_rag else rag_core.get_response(text))
                
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
                error_data = json.dumps({"error": str(e)}) + "\n"
                await response.write(error_data.encode('utf-8'))
            finally:
                await response.write_eof()

        # 执行流式处理
        await stream_pipeline(transcribed_text)
        return response
        
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
    app.router.add_post("/vision/describe", describe_image)