# /workspace/LiveTalking/api/avatar_api.py
# 数字人API路由 - 实现RESTful接口

import asyncio
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional
import logging

from aiohttp import web, multipart
from aiohttp.web import Request, Response

# 导入数字人相关模块
from pkg.avatars.generator_musetalk import get_generator
from pkg.avatars.session_manager import get_session_manager

# 导入日志记录器
from logger import logger


# 全局实例
_generator = None
_session_manager = None

def get_avatar_generator():
    """获取数字人生成器实例"""
    global _generator
    if _generator is None:
        _generator = get_generator()
    return _generator

def get_avatar_session_manager():
    """获取会话管理器实例"""
    global _session_manager
    if _session_manager is None:
        _session_manager = get_session_manager()
    return _session_manager

async def create_avatar(request: Request) -> Response:
    """
    POST /api/avatars - 创建数字人

    表单(multipart/form-data):
    - file: 必填，图片序列ZIP/单图/视频
    - avatar_id: 可选，不填则服务端生成

    返回：202 Accepted {avatar_id, job_id, status: PENDING}
    """
    try:
        # 1) 解析表单
        reader = await request.multipart()
        file_data = None
        filename = None
        avatar_id = None

        async for field in reader:
            if field.name == 'file':
                file_data = await field.read()
                filename = field.filename
            elif field.name == 'avatar_id':
                avatar_id = await field.text()

        if not file_data:
            return web.json_response({"error": "缺少必需的文件参数"}, status=400)

        # 2) 将临时文件“落到持久目录”，避免异步任务还未读就被删
        #    —— 修复点：不再用 NamedTemporaryFile + finally 删除
        from pathlib import Path
        import time, os, re

        base_dir = Path(__file__).resolve().parents[1]  # /workspace/LiveTalking/api -> /workspace/LiveTalking
        uploads_dir = base_dir / "data" / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # 简单清理 filename，避免奇怪字符
        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename or f'upload_{int(time.time())}')
        ts = int(time.time() * 1000)
        persistent_path = uploads_dir / f"{ts}_{safe_name}"

        with open(persistent_path, "wb") as f:
            f.write(file_data)

        # 3) 交给生成器（这里传的是“持久路径”）
        generator = get_avatar_generator()
        result = await generator.create_avatar(str(persistent_path), avatar_id)

        # 4) 立刻响应任务受理
        return web.json_response(result, status=202)

        # 5) 注意：文件清理交给生成器侧在任务完成后自行决定（或做后台定期清理）
    except Exception as e:
        logger.error(f"创建数字人失败: {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)
        
async def list_avatars(request: Request) -> Response:
    """
    GET /api/avatars - 获取数字人列表
    
    查询参数：
    - page: 页码（默认1）
    - page_size: 每页大小（默认20）
    
    返回：{avatars: [...], total: int, page: int, page_size: int}
    """
    try:
        # 获取查询参数
        page = int(request.query.get('page', 1))
        page_size = int(request.query.get('page_size', 20))
        
        # 获取生成器实例
        generator = get_avatar_generator()
        
        # 获取数字人列表
        result = generator.list_avatars(page, page_size)
        
        return web.json_response(result)
        
    except Exception as e:
        logger.error(f"获取数字人列表失败: {e}", exc_info=True)
        return web.json_response(
            {"error": str(e)},
            status=500
        )

async def get_avatar_detail(request: Request) -> Response:
    """
    GET /api/avatars/{id} - 获取数字人详情
    
    返回：数字人元数据与状态
    """
    try:
        avatar_id = request.match_info['id']
        
        # 获取生成器实例
        generator = get_avatar_generator()
        
        # 获取数字人信息
        avatar_info = generator.get_avatar_info(avatar_id)
        
        if not avatar_info:
            return web.json_response(
                {"error": f"数字人不存在: {avatar_id}"},
                status=404
            )
        
        return web.json_response(avatar_info)
        
    except Exception as e:
        logger.error(f"获取数字人详情失败: {e}", exc_info=True)
        return web.json_response(
            {"error": str(e)},
            status=500
        )

async def delete_avatar(request: Request) -> Response:
    """
    DELETE /api/avatars/{id} - 删除数字人 (增加占用保护)
    
    若被会话占用：返回409 Conflict并给出占用的session_id
    """
    try:
        avatar_id = request.match_info['id']
        generator = get_avatar_generator()
        
        # 1. 通过会话管理器检查是否被会话占用
        is_in_use, session_id = generator.is_avatar_in_use(avatar_id)
        if is_in_use:
            return web.json_response(
                {
                    "error": f"数字人 '{avatar_id}' 正在被会话 '{session_id}' 使用，无法删除。",
                    "session_id": session_id
                },
                status=409  # 409 Conflict 更符合语义
            )
        
        # 2. 删除数字人
        success = generator.delete_avatar(avatar_id)
        
        if not success:
            return web.json_response(
                {"error": f"数字人不存在或删除失败: {avatar_id}"},
                status=404
            )
        
        return web.json_response(
            {"message": f"数字人删除成功: {avatar_id}"},
            status=200
        )
        
    except Exception as e:
        logger.error(f"删除数字人失败: {e}", exc_info=True)
        return web.json_response(
            {"error": str(e)},
            status=500
        )

async def get_avatar_status(request: Request) -> Response:
    """
    GET /api/avatars/{id}/status - 获取数字人生成状态
    
    返回：{status: PENDING|SUCCEEDED|FAILED, error_code?: int, stderr_tail?: str}
    """
    try:
        avatar_id = request.match_info['id']
        
        # 获取生成器实例
        generator = get_avatar_generator()
        
        # 获取数字人信息
        avatar_info = generator.get_avatar_info(avatar_id)
        
        if not avatar_info:
            return web.json_response(
                {"error": f"数字人不存在: {avatar_id}"},
                status=404
            )
        
        # 构建状态响应
        status_response = {
            "status": avatar_info["status"]
        }
        
        # 如果是失败状态，添加错误信息
        if avatar_info["status"] == "FAILED":
            # 暂时返回基本信息
            status_response["error_code"] = -1
            status_response["stderr_tail"] = "生成失败"
        
        return web.json_response(status_response)
        
    except Exception as e:
        logger.error(f"获取数字人状态失败: {e}", exc_info=True)
        return web.json_response(
            {"error": str(e)},
            status=500
        )

async def get_avatar_preview(request: Request) -> Response:
    """
    GET /api/avatars/{id}/preview - 获取数字人预览图
    
    返回：第一帧图片
    """
    try:
        avatar_id = request.match_info['id']
        
        # 获取生成器实例
        generator = get_avatar_generator()
        
        # 获取数字人信息
        avatar_info = generator.get_avatar_info(avatar_id)
        
        if not avatar_info:
            return web.json_response(
                {"error": f"数字人不存在: {avatar_id}"},
                status=404
            )
        
        # 查找第一帧图片
        avatar_dir = generator.data_dir / avatar_id / "full_imgs"
        if not avatar_dir.exists():
            return web.json_response(
                {"error": "数字人图片不存在"},
                status=404
            )
        
        # 获取第一帧图片
        image_files = sorted(list(avatar_dir.glob("*.png")) + list(avatar_dir.glob("*.jpg")))
        if not image_files:
            return web.json_response(
                {"error": "数字人图片不存在"},
                status=404
            )
        
        first_image = image_files[0]
        
        # 返回图片文件
        return web.FileResponse(
            path=str(first_image),
            headers={
                'Content-Type': 'image/png', # 简单处理，可根据后缀名变化
                'Cache-Control': 'public, max-age=3600'
            }
        )
        
    except Exception as e:
        logger.error(f"获取数字人预览失败: {e}", exc_info=True)
        return web.json_response(
            {"error": str(e)},
            status=500
        )

async def switch_session_avatar(request: 'Request') -> 'Response':
    """
    PATCH /api/sessions/{session_id}/avatar
    请求体: {"avatar_id": "kelong"} 或表单 "avatar_id=kelong"

    中文注释（改造点）：
    - 不修改 musereal.py：保持原有推理线程逻辑不变
    - 采用“就地更新（in-place）”策略，把 MuseReal 实例里各循环列表的内容替换为新头像素材
      这样线程里持有的旧引用仍指向同一个 list 对象，但数据已经是新头像，热切换立即生效
    - 清空输出队列并重置 idx，避免旧帧残留；不清 render_event，防止误停线程
    """

    # 小工具：对 list-like 目标做就地更新（尽量兼容）
    def _inplace_update(dst_list, src_list):
        """中文注释：把 dst_list 内容改成 src_list，保持 dst_list 对象不变"""
        # 兼容普通list与支持clear/extend的自定义容器
        if hasattr(dst_list, "clear") and hasattr(dst_list, "extend"):
            dst_list.clear()
            dst_list.extend(src_list)
            return True
        # 退路：切片整体赋值
        try:
            dst_list[:] = src_list
            return True
        except Exception as e:
            logger.warning(f"in-place 更新失败，fallback: {e}")
            return False

    try:
        session_id = request.match_info['session_id']

        # 兼容 json / x-www-form-urlencoded
        try:
            payload = await request.json()
        except Exception:
            payload = await request.post()
        new_avatar_id = payload.get("avatar_id")
        if not new_avatar_id:
            return web.json_response({"error": "缺少 avatar_id"}, status=400)

        app = request.app
        opt = app.get('opt')
        if not opt:
            return web.json_response({"error": "服务未正确初始化（缺少opt）"}, status=500)

        # 1) 找到会话对应推理实例
        nerf_instance = app['nerfreals'].get(int(session_id))
        if not nerf_instance:
            return web.json_response({"error": "会话推理实例不存在"}, status=404)

        # 2) 兜底：尝试中断当前说话/tts，减少切换期间的拉扯
        try:
            if hasattr(nerf_instance, "flush_talk"):
                nerf_instance.flush_talk()
        except Exception:
            pass

        # 3) 根据模型选择正确的 loader 加载新素材（IO/CPU 走线程池，避免阻塞）
        model_name = getattr(opt, "model", "musetalk")
        if model_name == "musetalk":
            from musereal import load_avatar as model_load_avatar  # :contentReference[oaicite:0]{index=0}
        elif model_name == "lightreal":
            from lightreal import load_avatar as model_load_avatar
        elif model_name == "lipreal":
            from lipreal import load_avatar as model_load_avatar
        else:
            return web.json_response({"error": f"未知模型: {model_name}"}, status=400)

        loop = asyncio.get_event_loop()
        new_assets = await loop.run_in_executor(None, lambda: model_load_avatar(new_avatar_id))  # :contentReference[oaicite:1]{index=1}

        # 4) 关键：就地更新，而不是替换对象引用
        if model_name == "musetalk":
            # musereal.load_avatar 返回 5 元组：frame/mask/coord/mask_coords/latents :contentReference[oaicite:2]{index=2}
            (new_frames,
             new_masks,
             new_coords,
             new_mask_coords,
             new_latents) = new_assets

            ok = True
            # 以下对象在推理线程中被使用，必须 in-place 才能被旧引用感知
            if hasattr(nerf_instance, "frame_list_cycle"):
                ok &= _inplace_update(nerf_instance.frame_list_cycle, new_frames)
            if hasattr(nerf_instance, "mask_list_cycle"):
                ok &= _inplace_update(nerf_instance.mask_list_cycle, new_masks)
            if hasattr(nerf_instance, "coord_list_cycle"):
                ok &= _inplace_update(nerf_instance.coord_list_cycle, new_coords)
            if hasattr(nerf_instance, "mask_coords_list_cycle"):
                ok &= _inplace_update(nerf_instance.mask_coords_list_cycle, new_mask_coords)
            if hasattr(nerf_instance, "input_latent_list_cycle"):
                ok &= _inplace_update(nerf_instance.input_latent_list_cycle, new_latents)  # 推理线程按索引取用 :contentReference[oaicite:3]{index=3}

            # 如果有某个容器不是list导致 in-place 失败，做兜底：替换引用（兼容历史）
            if not ok:
                nerf_instance.frame_list_cycle = new_frames
                nerf_instance.mask_list_cycle = new_masks
                nerf_instance.coord_list_cycle = new_coords
                nerf_instance.mask_coords_list_cycle = new_mask_coords
                nerf_instance.input_latent_list_cycle = new_latents

        else:
            # 其他模型保持原逻辑（如有需要也可参照上面 in-place）
            # ……略……
            pass

        # 5) 清空产出队列，重置游标，避免旧帧残留（不触碰 render_event）
        try:
            if hasattr(nerf_instance, "res_frame_queue"):
                while not nerf_instance.res_frame_queue.empty():
                    nerf_instance.res_frame_queue.get_nowait()
        except Exception:
            pass
        if hasattr(nerf_instance, "idx"):
            nerf_instance.idx = 0

        # 6) 会话-数字人占用关系更新（若你项目里有这个管理器）
        try:
            from pkg.avatars.session_manager import get_session_manager as get_avatar_session_manager
            sm = get_avatar_session_manager()
            sm.switch_avatar(session_id, new_avatar_id)  # :contentReference[oaicite:4]{index=4}
        except Exception:
            pass

        return web.json_response({
            "ok": True,
            "session_id": session_id,
            "avatar_id": new_avatar_id,
            "model": model_name
        })
    except Exception as e:
        logger.error(f"切换数字人失败: {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)

async def get_task_status(request: Request) -> Response:
    """
    GET /api/avatars/tasks/{job_id} - 获取任务状态
    
    返回：任务状态信息
    """
    try:
        job_id = request.match_info['job_id']
        
        # 获取生成器实例
        generator = get_avatar_generator()
        
        # 获取任务状态
        task_info = generator.get_task_status(job_id)
        
        if not task_info:
            return web.json_response(
                {"error": f"任务不存在: {job_id}"},
                status=404
            )
        
        return web.json_response(task_info)
        
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}", exc_info=True)
        return web.json_response(
            {"error": str(e)},
            status=500
        )

def register_avatar_routes(app: web.Application):
    """
    注册数字人相关路由
    
    Args:
        app: aiohttp应用实例
    """
    # 数字人管理路由
    app.router.add_post('/api/avatars', create_avatar)
    app.router.add_get('/api/avatars', list_avatars)
    app.router.add_get('/api/avatars/{id}', get_avatar_detail)
    app.router.add_delete('/api/avatars/{id}', delete_avatar)
    app.router.add_get('/api/avatars/{id}/status', get_avatar_status)
    app.router.add_get('/api/avatars/{id}/preview', get_avatar_preview)
    
    # 任务状态路由
    app.router.add_get('/api/avatars/tasks/{job_id}', get_task_status)
    
    # 会话数字人切换路由
    app.router.add_patch('/api/sessions/{session_id}/avatar', switch_session_avatar)
    # 兼容某些前端写法用 POST 切换
    app.router.add_post('/api/sessions/{session_id}/avatar', switch_session_avatar)
    
    logger.info("数字人API路由注册完成")
