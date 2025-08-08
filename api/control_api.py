# /workspace/LiveTalking/api/control_api.py
# -*- 简化注释：导入必要的库 -*-
import asyncio
import gc
import json

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from api.utils import build_nerfreal, randN
from logger import logger
from webrtc import HumanPlayer


async def offer(request):
    """
    处理WebRTC的offer请求，建立P2P连接并创建一个新的虚拟人会话。
    Handles WebRTC offer requests, establishes a P2P connection, and creates a new virtual human session.
    """
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    sessionid = randN(6)
    
    # 从应用上下文中获取所需对象
    app = request.app
    opt = app['opt']
    model = app['model']
    avatar = app['avatar']
    nerfreals = app['nerfreals']
    pcs = app['pcs']
    
    nerfreals[sessionid] = None
    logger.info('创建会话 sessionid=%d, 当前会话数=%d', sessionid, len(nerfreals))

    # 配置ICE服务器
    ice_servers = [RTCIceServer(urls="stun:stun.miwifi.com:3478")]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        """处理连接状态变化，用于清理和关闭会话"""
        logger.info("连接状态变为: %s", pc.connectionState)
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            if sessionid in nerfreals:
                logger.info(f"因连接状态为 {pc.connectionState}，正在关闭会话 {sessionid}")
                # 弹出并删除虚拟人实例，释放资源
                session_to_close = nerfreals.pop(sessionid, None)
                if session_to_close:
                    del session_to_close
                gc.collect()
            if pc in pcs:
                await pc.close()
                pcs.discard(pc)

    logger.info(f"会话 {sessionid}: 开始构建虚拟人实例...")
    # 在线程池中执行耗时的虚拟人构建操作
    nerfreal = await asyncio.get_event_loop().run_in_executor(
        None, build_nerfreal, opt, model, avatar, sessionid
    )
    logger.info(f"会话 {sessionid}: 虚拟人实例构建完成。")

    # 检查会话在实例创建完成前是否已被关闭
    if sessionid not in nerfreals:
        logger.warning(f"会话 {sessionid} 在实例准备好之前已关闭。")
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"code": -1, "msg": "连接在设置过程中失败"})
        )
    nerfreals[sessionid] = nerfreal

    # 创建播放器并添加音视频轨道
    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.audio)
    pc.addTrack(player.video)
    
    # 设置视频编码器偏好 (H264/VP8)
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

    # 设置远程和本地SDP描述
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid})
    )


async def interrupt_talk(request):
    """
    中断指定会话的虚拟人当前语音播报。
    Interrupts the current speech of the virtual human for a given session.
    """
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        nerfreals = request.app['nerfreals']
        
        if sessionid in nerfreals:
            nerfreals[sessionid].flush_talk() # 调用虚拟人实例的清空语音队列方法
            
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
    except Exception as e:
        logger.exception('interrupt_talk接口异常:')
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"code": -1, "msg": str(e)})
        )


async def set_audiotype(request):
    """
    设置音频类型或重新初始化虚拟人状态。
    Sets the audio type or reinitializes the virtual human's state.
    """
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        nerfreals = request.app['nerfreals']
        
        if sessionid in nerfreals:
            nerfreals[sessionid].set_custom_state(params['audiotype'], params['reinit'])
            
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
    except Exception as e:
        logger.exception('set_audiotype接口异常:')
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"code": -1, "msg": str(e)})
        )


async def record(request):
    """
    控制录制功能的开始和结束。
    Controls the start and end of the recording feature.
    """
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        nerfreals = request.app['nerfreals']

        if sessionid in nerfreals:
            # 根据请求类型调用开始或停止录制
            if params['type'] == 'start_record':
                nerfreals[sessionid].start_recording()
            elif params['type'] == 'end_record':
                nerfreals[sessionid].stop_recording()
                
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
    except Exception as e:
        logger.exception('record接口异常:')
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"code": -1, "msg": str(e)})
        )


async def is_speaking(request):
    """
    查询指定会话的虚拟人当前是否正在讲话。
    Queries if the virtual human of a given session is currently speaking.
    """
    params = await request.json()
    sessionid = int(params.get('sessionid', 0))
    nerfreals = request.app['nerfreals']
    
    speaking = nerfreals[sessionid].is_speaking() if sessionid in nerfreals else False
    
    return web.Response(content_type="application/json", text=json.dumps({"code": 0, "data": speaking}))


def register_control_routes(app):
    """
    将WebRTC和控制相关的路由注册到 aiohttp 应用。
    Registers WebRTC and control related routes to the aiohttp application.
    """
    app.router.add_post("/offer", offer)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/is_speaking", is_speaking)

