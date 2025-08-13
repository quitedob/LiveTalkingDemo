# /workspace/LiveTalking/api/rag_api.py
import asyncio
import json
from pathlib import Path

from aiohttp import web

from logger import logger
from pkg.rag.config import (KB_NAME_PATTERN, MAX_KBS, TASK_STATUS_POLL_INTERVAL_S)
from pkg.rag.file_processor import FileProcessor
from pkg.rag.knowledge_base import KnowledgeBase
from pkg.rag.rag_core import RAGCore

async def setup_rag_app(app):
    """
    在 aiohttp 应用启动时初始化 RAG 相关的实例。
    """
    logger.info("正在初始化RAG系统...")
    rag_core = RAGCore(app['llm_client'])
    app['rag_core'] = rag_core
    app['tasks'] = {}
    logger.info("RAG系统初始化完成。")

async def create_kb(request):
    """
    创建新的知识库。
    """
    try:
        data = await request.post()
        kb_name = data.get('kb_name')
        file_field = data.get('file')

        if not kb_name or not KB_NAME_PATTERN.match(kb_name):
            return web.json_response({"error": "知识库名称不合法"}, status=400)
        
        current_kbs = KnowledgeBase.list_kbs()
        if kb_name in current_kbs:
            return web.json_response({"error": f"知识库 '{kb_name}' 已存在"}, status=409)
        if len(current_kbs) >= MAX_KBS:
            return web.json_response({"error": f"已达到最大知识库数量"}, status=400)
        if not file_field:
            return web.json_response({"error": "缺少上传文件"}, status=400)

        original_filename = file_field.filename
        file_bytes = file_field.file.read()
        
        task_id = f"upload_{kb_name}_{original_filename}"
        request.app['tasks'][task_id] = {"status": "processing", "message": "开始处理文件..."}
        
        async def process_task():
            try:
                txt_path = await FileProcessor.save_and_process_file(file_bytes, original_filename, kb_name)
                request.app['tasks'][task_id]['message'] = "文本提取完成，正在向量化..."
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    lambda: KnowledgeBase(kb_name).add_txt_file(txt_path, original_filename)
                )
                
                request.app['tasks'][task_id] = {"status": "success", "message": "知识库创建成功"}
                if txt_path.exists():
                    txt_path.unlink()
            except Exception as e:
                logger.error(f"创建知识库 '{kb_name}' 后台任务失败: {e}", exc_info=True)
                clean_error = str(e).replace('\n', ' ').replace('\r', '')
                request.app['tasks'][task_id] = {"status": "failed", "message": clean_error}

        asyncio.create_task(process_task())
        return web.json_response({"message": "文件上传成功，后台处理中...", "task_id": task_id}, status=202)

    except Exception as e:
        logger.exception("创建知识库API异常")
        return web.json_response({"error": f"服务器内部错误: {e}"}, status=500)

async def delete_kb(request):
    kb_name = request.match_info.get('name')
    try:
        KnowledgeBase.delete_kb(kb_name)
        await FileProcessor.cleanup_kb_files(kb_name)
        rag_core = request.app['rag_core']
        if rag_core.current_kb_name == kb_name:
            rag_core.switch_kb(None)
        return web.json_response({"message": f"知识库 '{kb_name}' 已成功删除"}, status=200)
    except Exception as e:
        logger.exception(f"删除知识库 '{kb_name}' API异常")
        clean_error = str(e).replace('\n', ' ').replace('\r', '')
        return web.json_response({"error": f"删除失败: {clean_error}"}, status=500)

async def switch_kb(request):
    kb_name = request.match_info.get('name')
    rag_core = request.app['rag_core']
    if rag_core.switch_kb(kb_name):
        return web.json_response({"message": f"知识库已切换为 '{kb_name}'"}, status=200)
    else:
        return web.json_response({"error": f"知识库 '{kb_name}' 不存在"}, status=404)

async def set_rag_mode(request):
    try:
        data = await request.json()
        use_rag = data.get('use_rag', True)
        if not isinstance(use_rag, bool):
            return web.json_response({"error": "参数 'use_rag' 必须是布尔值"}, status=400)
        request.app['rag_core'].set_rag_mode(use_rag)
        return web.json_response({"message": f"RAG模式已设置为 {use_rag}"}, status=200)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def update_prompt(request):
    try:
        data = await request.json()
        prompt = data.get('prompt')
        if not prompt or not isinstance(prompt, str):
            return web.json_response({"error": "参数 'prompt' 不能为空或无效"}, status=400)
        request.app['rag_core'].update_system_prompt(prompt)
        return web.json_response({"message": "系统提示词更新成功"}, status=200)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def get_task_status(request):
    task_id = request.match_info.get('task_id')
    task_info = request.app['tasks'].get(task_id)
    if not task_info:
        return web.json_response({"error": "任务ID不存在"}, status=404)
    return web.json_response(task_info)

async def rag_chat(request):
    try:
        data = await request.json()
        query = data.get('query')
        if not query:
            return web.json_response({"error": "缺少查询 'query'"}, status=400)
        rag_core = request.app['rag_core']
        response = web.StreamResponse(headers={'Content-Type': 'application/json; charset=utf-8'})
        await response.prepare(request)
        async for chunk in rag_core.get_response(query):
            await response.write(json.dumps({"response": chunk}).encode('utf-8') + b'\n')
        await response.write_eof()
        return response
    except Exception as e:
        logger.exception("RAG聊天API异常")
        return web.Response(status=500, text=f"服务器内部错误: {e}")

async def list_kbs(request):
    kbs = KnowledgeBase.list_kbs()
    return web.json_response({"knowledge_bases": kbs}, status=200)

async def get_config(request):
    rag_core = request.app['rag_core']
    config = {
        "use_rag": rag_core.use_rag,
        "current_kb": rag_core.current_kb_name,
        "system_prompt": rag_core.system_prompt
    }
    return web.json_response(config, status=200)

def register_rag_routes(app):
    app.on_startup.append(setup_rag_app)
    app.router.add_post("/kb/create", create_kb)
    app.router.add_delete("/kb/delete/{name}", delete_kb)
    app.router.add_post("/kb/switch/{name}", switch_kb)
    app.router.add_get("/kb/list", list_kbs)
    app.router.add_post("/config/rag_mode", set_rag_mode)
    app.router.add_post("/config/prompt", update_prompt)
    app.router.add_get("/config/get", get_config)
    app.router.add_get("/kb/status/{task_id}", get_task_status)
    app.router.add_post("/rag/chat", rag_chat)
    logger.info("RAG API 路由已注册。")
