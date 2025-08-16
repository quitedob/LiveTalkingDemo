# /workspace/LiveTalking/api/control_api.py
# -*- coding: utf-8 -*-
# 目标：支持 WebRTC 断线重连与会话宽限，不把 disconnected 当作立即结束
# 说明：采纳您的方案进行重构，代码逻辑清晰，稳定性高。

import asyncio
import gc
import json
import time
from typing import Dict, Optional

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from api.utils import build_nerfreal, randN
from logger import logger
from webrtc import HumanPlayer
# 中文注释：引入会话管理器，统一登记/反登记，打通“热切换”接口与实际运行会话
from pkg.avatars.session_manager import get_session_manager


# =============== 配置区（可按需调整） ===============
GRACE_TTL_SECONDS = 120      # 中文注释：断线状态下，会话保留的宽限期（秒）
ICE_SERVERS = [
    RTCIceServer(urls="stun:stun.miwifi.com:3478"),
    # 如需公网更稳，建议加 Google STUN 或自己的 TURN
    # RTCIceServer(urls="stun:stun.l.google.com:19302"),
    # RTCIceServer(urls=["turns:your.turn.server:5349"], username="xxx", credential="yyy"),
]
# ====================================================


def _ensure_runtime_buckets(app: web.Application):
    """中文注释：确保本模块需要的运行时容器存在（懒创建）。"""
    if 'session_pcs' not in app:
        app['session_pcs']: Dict[int, RTCPeerConnection] = {}
    if 'disconnect_deadline' not in app:
        app['disconnect_deadline']: Dict[int, float] = {}
    if 'disconnect_tasks' not in app:
        app['disconnect_tasks']: Dict[int, asyncio.Task] = {}


def _mark_deadline(app: web.Application, sessionid: int, seconds: int = GRACE_TTL_SECONDS):
    """中文注释：记录会话的“存活截止时间”，用于宽限期内可重连。"""
    _ensure_runtime_buckets(app)
    app['disconnect_deadline'][sessionid] = time.time() + max(1, seconds)


def _clear_deadline(app: web.Application, sessionid: int):
    """中文注释：清理会话的截止时间与定时任务。"""
    _ensure_runtime_buckets(app)
    app['disconnect_deadline'].pop(sessionid, None)
    task = app['disconnect_tasks'].pop(sessionid, None)
    if task and not task.done():
        task.cancel()


async def _schedule_disconnect_cleanup(app: web.Application, sessionid: int):
    """中文注释：在宽限期到期后清理会话（若仍未重连）。"""
    _ensure_runtime_buckets(app)
    # 先取消旧任务，避免重复
    task_to_cancel = app['disconnect_tasks'].pop(sessionid, None)
    if task_to_cancel and not task_to_cancel.done():
        task_to_cancel.cancel()
    
    _mark_deadline(app, sessionid)

    async def _job():
        try:
            deadline = app['disconnect_deadline'].get(sessionid)
            if not deadline:
                return
            
            await asyncio.sleep(deadline - time.time())

            if time.time() < app.get('disconnect_deadline', {}).get(sessionid, float('inf')):
                # 中文注释：如果截止时间被更新（比如心跳），则此任务作废
                return

        except (asyncio.CancelledError, TypeError):
            return

        # 到期后执行真正清理
        nerfreals = app['nerfreals']
        pcs = app['pcs']
        session_pc = app['session_pcs'].pop(sessionid, None)
        
        # 中文注释：关键：从 SessionManager 里反登记
        sm = get_session_manager()
        sm.close_session(str(sessionid))
        
        if session_pc:
            try:
                if session_pc in pcs:
                    await session_pc.close()
                    pcs.discard(session_pc)
            except Exception:
                pass

        if sessionid in nerfreals:
            logger.info("会话 %s 宽限到期，释放虚拟人实例", sessionid)
            session_to_close = nerfreals.pop(sessionid, None)
            if session_to_close:
                try:
                    del session_to_close
                except Exception:
                    pass
                gc.collect()

        _clear_deadline(app, sessionid)
        logger.info("会话 %s 已在宽限到期后彻底清理", sessionid)

    task = asyncio.create_task(_job())
    app['disconnect_tasks'][sessionid] = task


async def _attach_tracks(pc: RTCPeerConnection, nerfreal) -> None:
    """中文注释：把 HumanPlayer 的音视频轨道挂到新的 PeerConnection 上。"""
    player = HumanPlayer(nerfreal)
    pc.addTrack(player.audio)
    pc.addTrack(player.video)

    # 中文注释：设置视频编码器偏好（H264/VP8）以提高兼容性
    try:
        capabilities = RTCRtpSender.getCapabilities("video")
        codecs_list = getattr(capabilities, "codecs", None) or capabilities.get("codecs", [])
        if codecs_list:
            preferences = [c for c in codecs_list if c.mimeType.upper().endswith(("H264", "VP8"))]
            video_transceiver = next(
                (t for t in pc.getTransceivers() if t.sender and t.sender.track and t.sender.track.kind == "video"),
                None
            )
            if video_transceiver and preferences:
                video_transceiver.setCodecPreferences(preferences)
                logger.info("已成功设置视频首选编码器为: %s", [c.mimeType for c in preferences])
    except Exception as e:
        logger.warning("设置编解码器偏好失败: %s", e)


async def _new_pc(app: web.Application, sessionid: int) -> RTCPeerConnection:
    """中文注释：创建一个新的 RTCPeerConnection 并注册状态监听。"""
    pcs = app['pcs']
    nerfreals = app['nerfreals']

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ICE_SERVERS))
    pcs.add(pc)
    app['session_pcs'][sessionid] = pc

    @pc.on("connectionstatechange")
    async def _on_state_change():
        state = pc.connectionState
        logger.info("连接状态变为: %s", state)
        if state == "connected":
            # 一旦连上，清掉任何“断线清理”的倒计时
            _clear_deadline(app, sessionid)
        elif state == "disconnected":
            # 中文注释：断网/弱网/切网下常见，给宽限期，不要立即清理
            logger.info("会话 %s 进入 disconnected，启动宽限计时 %ss", sessionid, GRACE_TTL_SECONDS)
            await _schedule_disconnect_cleanup(app, sessionid)
        elif state in ("failed", "closed"):
            # 中文注释：失败或关闭：关闭当前 PC，但暂留 nerfreal 以便重连
            try:
                if pc in pcs:
                    await pc.close()
                    pcs.discard(pc)
            finally:
                # 进入宽限：允许客户端发 /reconnect 来接回会话
                await _schedule_disconnect_cleanup(app, sessionid)

    # 若会话存在，挂上轨道
    if sessionid in nerfreals and nerfreals[sessionid] is not None:
        await _attach_tracks(pc, nerfreals[sessionid])

    return pc


async def offer(request: web.Request) -> web.Response:
    """
    中文注释：处理首次 WebRTC offer（建立全新会话），返回 answer + sessionid
    """
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    app = request.app
    _ensure_runtime_buckets(app)

    # 生成新的会话 id
    sessionid = randN(6)

    opt = app['opt']
    model = app['model']
    avatar = app['avatar']
    nerfreals = app['nerfreals']

    # 中文注释：为热切换准备：获取初始 avatar_id
    initial_avatar_id = str(opt.avatar_id) if hasattr(opt, 'avatar_id') else None

    nerfreals[sessionid] = None
    logger.info('创建会话 sessionid=%d, 当前会话数=%d', sessionid, len(nerfreals))

    # 构建虚拟人（避免阻塞事件循环）
    logger.info("会话 %s: 开始构建虚拟人实例...", sessionid)
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, opt, model, avatar, sessionid)
    logger.info("会话 %s: 虚拟人实例构建完成。", sessionid)
    nerfreals[sessionid] = nerfreal

    # 中文注释：关键：将会话登记到 SessionManager（注意使用 str 作为 key）
    sm = get_session_manager()
    sm.create_session(str(sessionid), initial_avatar_id)

    # 新建 PC 并挂轨
    pc = await _new_pc(app, sessionid)

    # 常规 SDP 交换
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "sessionid": sessionid
    })


async def reconnect(request: web.Request) -> web.Response:
    """
    中文注释：处理重连/ICE 重启。
    - 输入：{ sessionid, sdp, type } 其中 sdp/type 是新的 offer（可带 iceRestart）
    - 行为：若旧 PC 存在则关闭，创建新 PC，复用原 nerfreal，返回新的 answer
    """
    params = await request.json()
    sessionid = int(params.get("sessionid", 0))
    if not sessionid:
        return web.json_response({"code": -1, "msg": "需要有效的 sessionid"}, status=400)

    app = request.app
    _ensure_runtime_buckets(app)
    nerfreals = app['nerfreals']

    if sessionid not in nerfreals:
        return web.json_response({"code": -1, "msg": "会话不存在或已过期"}, status=404)

    # 关闭旧 PC（若有），但不销毁会话
    old_pc: Optional[RTCPeerConnection] = app['session_pcs'].pop(sessionid, None)
    if old_pc:
        try:
            await old_pc.close()
        except Exception:
            pass
        if old_pc in app['pcs']:
            app['pcs'].discard(old_pc)

    # 新建 PC，挂轨
    pc = await _new_pc(app, sessionid)

    # 套上新的 SDP（通常来自 createOffer({iceRestart:true}))
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # 重连成功→取消宽限清理
    _clear_deadline(app, sessionid)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "sessionid": sessionid
    })


async def session_heartbeat(request: web.Request) -> web.Response:
    """
    中文注释：前端心跳续期。调用即可把会话宽限期往后顺延。
    - 输入：{ sessionid }
    """
    params = await request.json()
    sessionid = int(params.get("sessionid", 0))
    if not sessionid:
        return web.json_response({"code": -1, "msg": "需要有效的 sessionid"}, status=400)

    if sessionid not in request.app['nerfreals']:
        return web.json_response({"code": -1, "msg": "会话不存在或已过期"}, status=404)

    _mark_deadline(request.app, sessionid)
    return web.json_response({"code": 0, "msg": "ok", "ttl": GRACE_TTL_SECONDS})


async def interrupt_talk(request: web.Request) -> web.Response:
    """中文注释：打断指定会话的 TTS 播报。"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        nerfreals = request.app['nerfreals']

        if sessionid in nerfreals:
            nerfreals[sessionid].flush_talk()

        return web.json_response({"code": 0, "msg": "ok"})
    except Exception as e:
        logger.exception('interrupt_talk 接口异常:')
        return web.json_response({"code": -1, "msg": str(e)}, status=500)


async def set_audiotype(request: web.Request) -> web.Response:
    """中文注释：设置自定义待机音频/视频类型。"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        nerfreals = request.app['nerfreals']

        if sessionid in nerfreals:
            nerfreals[sessionid].set_custom_state(params['audiotype'], params.get('reinit', True))

        return web.json_response({"code": 0, "msg": "ok"})
    except Exception as e:
        logger.exception('set_audiotype 接口异常:')
        return web.json_response({"code": -1, "msg": str(e)}, status=500)


async def record(request: web.Request) -> web.Response:
    """中文注释：开始/结束录制。"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        nerfreals = request.app['nerfreals']

        if sessionid in nerfreals:
            if params['type'] == 'start_record':
                nerfreals[sessionid].start_recording()
            elif params['type'] == 'end_record':
                nerfreals[sessionid].stop_recording()

        return web.json_response({"code": 0, "msg": "ok"})
    except Exception as e:
        logger.exception('record 接口异常:')
        return web.json_response({"code": -1, "msg": str(e)}, status=500)


async def is_speaking(request: web.Request) -> web.Response:
    """中文注释：查询当前是否在说话。"""
    params = await request.json()
    sessionid = int(params.get('sessionid', 0))
    nerfreals = request.app['nerfreals']
    speaking = nerfreals[sessionid].is_speaking() if sessionid in nerfreals else False
    return web.json_response({"code": 0, "data": speaking})


def register_control_routes(app: web.Application):
    """
    中文注释：注册路由。新增 /reconnect 与 /session/heartbeat
    """
    app.router.add_post("/offer", offer)
    app.router.add_post("/reconnect", reconnect)              # 新增：重连/ICE 重启
    app.router.add_post("/session/heartbeat", session_heartbeat)  # 新增：心跳续期
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/is_speaking", is_speaking)