# /workspace/LiveTalking/api/rag_api.py
import asyncio
import json
from pathlib import Path

from aiohttp import web

from logger import logger
from pkg.rag.config import (KB_NAME_PATTERN, MAX_FILE_SIZE_MB, MAX_KBS,
                            TASK_STATUS_POLL_INTERVAL_S)
from pkg.rag.file_processor import FileProcessor
from pkg.rag.knowledge_base import KnowledgeBase
from pkg.rag.rag_core import RAGCore


async def setup_rag_app(app):
    """
    在 aiohttp 应用启动时初始化 RAG 相关的实例。
    """
    logger.info("正在初始化RAG系统...")
    # 初始化知识库管理器
    knowledge_base = KnowledgeBase()
    # 初始化RAG核心逻辑，并注入应用上下文中的共享llm_client
    rag_core = RAGCore(knowledge_base, app['llm_client'])
    
    # 将实例存储在app上下文中，方便后续的请求处理器访问
    app['kb_manager'] = knowledge_base
    app['rag_core'] = rag_core
    app['tasks'] = {} # 用于存储后台任务的状态
    logger.info("RAG系统初始化完成。")


# --- API 处理函数 ---

async def create_kb(request):
    """
    创建新的知识库。可以基于上传的文件或纯文本。
    """
    try:
        data = await request.post()
        kb_name = data.get('kb_name')
        file_field = data.get('file')

        kb_manager = request.app['kb_manager']
        
        # 1. 验证输入
        if not kb_name:
            return web.json_response({"error": "缺少知识库名称 'kb_name'"}, status=400)
        if not KB_NAME_PATTERN.match(kb_name):
            return web.json_response({"error": "知识库名称不合法，应为3-20位的英文或数字"}, status=400)
        if kb_name in kb_manager.list_kbs():
            return web.json_response({"error": f"知识库名称 '{kb_name}' 已存在"}, status=409)
        if len(kb_manager.list_kbs()) >= MAX_KBS:
            return web.json_response({"error": f"已达到最大知识库数量（{MAX_KBS}）"}, status=400)
        if not file_field:
            return web.json_response({"error": "缺少文件 'file'"}, status=400)

        # 文件大小验证由 aiohttp 的 client_max_size 全局处理

        # --- 关键修复：在API处理器返回前，立即读取文件内容 ---
        original_filename = file_field.filename
        file_bytes = file_field.file.read()
        
        task_id = f"upload_{kb_name}"
        request.app['tasks'][task_id] = {"status": "processing", "message": "开始处理文件..."}
        
        # 2. 异步执行文件处理和知识库添加
        async def process_task():
            try:
                # 将文件字节和原始文件名传递给后台任务
                txt_path = await FileProcessor.save_and_process_file(file_bytes, original_filename, kb_name)
                request.app['tasks'][task_id]['message'] = "文本提取完成，正在向量化..."
                
                # 将提取的文本添加到知识库
                await kb_manager.add_document(kb_name, txt_path)
                request.app['tasks'][task_id] = {"status": "success", "message": "知识库创建成功"}
            except Exception as e:
                logger.error(f"创建知识库 '{kb_name}' 的后台任务失败: {e}")
                request.app['tasks'][task_id] = {"status": "failed", "message": str(e)}

        # 将任务交由事件循环处理
        asyncio.create_task(process_task())

        # 3. 立即返回202 Accepted，并提供任务ID
        return web.json_response(
            {"message": "文件上传成功，后台处理中...", "task_id": task_id},
            status=202
        )

    except Exception as e:
        logger.exception("创建知识库API异常:")
        return web.json_response({"error": f"服务器内部错误: {e}"}, status=500)


async def delete_kb(request):
    """
    删除指定的知识库。
    """
    kb_name = request.match_info.get('name')
    kb_manager = request.app['kb_manager']
    
    try:
        # 1. 删除ChromaDB中的集合
        kb_manager.delete_kb(kb_name)
        
        # 2. 删除关联的文件
        await FileProcessor.cleanup_kb_files(kb_name)
        
        # 3. 如果删除的是当前激活的知识库，则清空
        rag_core = request.app['rag_core']
        if rag_core.current_kb == kb_name:
            rag_core.current_kb = None
            logger.info(f"当前激活的知识库 '{kb_name}' 已被删除，已重置。")

        return web.json_response({"message": f"知识库 '{kb_name}' 已成功删除"}, status=200)
    except Exception as e:
        logger.exception(f"删除知识库 '{kb_name}' API异常:")
        return web.json_response({"error": f"删除失败: {e}"}, status=500)


async def switch_kb(request):
    """
    切换当前对话使用的知识库。
    """
    kb_name = request.match_info.get('name')
    rag_core = request.app['rag_core']

    if rag_core.switch_kb(kb_name):
        return web.json_response({"message": f"知识库已切换为 '{kb_name}'"}, status=200)
    else:
        return web.json_response({"error": f"知识库 '{kb_name}' 不存在"}, status=404)


async def set_rag_mode(request):
    """
    设置RAG模式的开关。
    """
    try:
        data = await request.json()
        use_rag = data.get('use_rag')
        if not isinstance(use_rag, bool):
            return web.json_response({"error": "参数 'use_rag' 必须是布尔值"}, status=400)

        request.app['rag_core'].set_rag_mode(use_rag)
        return web.json_response({"message": f"RAG模式已设置为 {use_rag}"}, status=200)
    except Exception as e:
        logger.exception("设置RAG模式API异常:")
        return web.json_response({"error": str(e)}, status=500)


async def update_prompt(request):
    """
    更新系统的动态提示词。
    """
    try:
        data = await request.json()
        prompt = data.get('prompt')
        if not isinstance(prompt, str) or not prompt:
            return web.json_response({"error": "参数 'prompt' 必须是有效的非空字符串"}, status=400)

        if request.app['rag_core'].update_system_prompt(prompt):
            return web.json_response({"message": "系统提示词更新成功"}, status=200)
        else:
            return web.json_response({"error": "更新提示词时发生内部错误"}, status=500)

    except Exception as e:
        logger.exception("更新提示词API异常:")
        return web.json_response({"error": str(e)}, status=500)


async def get_task_status(request):
    """
    获取指定后台任务的状态。
    """
    task_id = request.match_info.get('task_id')
    task_info = request.app['tasks'].get(task_id)
    
    if not task_info:
        return web.json_response({"error": "任务ID不存在"}, status=404)
    
    response = web.json_response(task_info)
    # 如果任务仍在进行中，建议客户端在一段时间后再次查询
    if task_info.get('status') == 'processing':
        response.headers['Retry-After'] = str(TASK_STATUS_POLL_INTERVAL_S)
        
    return response


async def rag_chat(request):
    """
    处理RAG聊天请求，返回流式响应。
    """
    try:
        data = await request.json()
        query = data.get('query')
        if not query:
            return web.json_response({"error": "缺少查询内容 'query'"}, status=400)
            
        rag_core = request.app['rag_core']
        
        # 准备流式响应
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={'Content-Type': 'application/json; charset=utf-8'}, # 返回json stream
        )
        await response.prepare(request)

        # 异步获取并写入响应块
        async for chunk in rag_core.get_response(query):
            await response.write(json.dumps({"response": chunk}).encode('utf-8') + b'\n')
        
        await response.write_eof()
        return response

    except Exception as e:
        logger.exception("RAG聊天API异常:")
        # 对于流式API，异常时难以返回JSON错误，只能中断连接
        return web.Response(status=500, text=f"服务器内部错误: {e}")


async def list_kbs(request):
    """
    获取所有知识库的列表。
    """
    kb_manager = request.app['kb_manager']
    kbs = kb_manager.list_kbs()
    return web.json_response({"knowledge_bases": kbs}, status=200)

async def get_config(request):
    """
    获取当前RAG的配置状态。
    """
    rag_core = request.app['rag_core']
    config = {
        "use_rag": rag_core.use_rag,
        "current_kb": rag_core.current_kb,
        "system_prompt": rag_core.system_prompt
    }
    return web.json_response(config, status=200)

def register_rag_routes(app):
    """
    将所有RAG相关的路由注册到aiohttp应用，并设置启动时的初始化任务。
    """
    # 在应用启动时调用 setup_rag_app
    app.on_startup.append(setup_rag_app)
    
    # 注册API路由
    app.router.add_post("/kb/create", create_kb)
    app.router.add_delete("/kb/delete/{name}", delete_kb)
    app.router.add_post("/kb/switch/{name}", switch_kb)
    app.router.add_get("/kb/list", list_kbs) # 新增
    app.router.add_post("/config/rag_mode", set_rag_mode)
    app.router.add_post("/config/prompt", update_prompt)
    app.router.add_get("/config/get", get_config) # 新增
    app.router.add_get("/kb/status/{task_id}", get_task_status)
    app.router.add_post("/rag/chat", rag_chat) # 新增的聊天端点
    logger.info("RAG API 路由已注册。")
