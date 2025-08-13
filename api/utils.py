# /workspace/LiveTalking/api/utils.py
# -*- 简化注释：导入必要的库 -*-
import asyncio
import json
import os
import random
import tempfile
from functools import partial
from typing import Dict

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from logger import logger


def randN(N: int) -> int:
    """
    生成一个指定长度的随机整数。
    Generate a random integer of a specified length.
    """
    min_val = pow(10, N - 1)
    max_val = pow(10, N)
    return random.randint(min_val, max_val - 1)


def build_nerfreal(opt, model, avatar, sessionid: int):
    """
    根据配置构建并返回一个虚拟人实例。
    Build and return a virtual human instance based on the configuration.
    """
    opt.sessionid = sessionid
    nerfreal = None
    # 根据模型名称动态导入并实例化对应的虚拟人类
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt, model, avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt, model, avatar)
    elif opt.model == 'ernerf':
        # ernerf 模型暂未实现
        pass
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt, model, avatar)
    return nerfreal


async def transcribe_audio(asr_model, audio_bytes: bytes) -> str:
    """
    将音频字节流通过FunASR转录为文本。
    Transcribe audio byte stream to text using FunASR.
    """
    loop = asyncio.get_event_loop()
    tmp_path = None
    try:
        # 创建临时文件保存音频数据
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # 使用偏函数固定 FunASR 模型的一些参数
        generate_fn = partial(
            asr_model.generate,
            input=tmp_path,
            cache={},
            language="zn+en",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15
        )
        # 在线程池中执行同步的ASR模型推理
        res = await loop.run_in_executor(None, generate_fn)
        
        # 处理返回结果
        res = res[0] if isinstance(res, list) and res else res if isinstance(res, dict) else {}
        text_raw = res.get("text", "")
        
        # 对原始文本进行后处理
        if text_raw:
            processed_text = rich_transcription_postprocess(text_raw)
            logger.info(f"ASR 转写结果: {processed_text}")
            return processed_text
        return ""
    except Exception as e:
        logger.error(f"ASR转写出错: {e}")
        return ""
    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


async def post(url: str, data: str):
    """
    一个简单的异步POST请求函数。
    A simple asynchronous POST request function.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, timeout=30) as response:
                return await response.text()
    except Exception as e:
        logger.info(f'POST请求错误: {e}')
        return None


async def run_push_session(app, push_url: str, sessionid: int):
    """
    为rtcpush模式运行一个独立的虚拟人会话。
    Run a separate virtual human session for rtcpush mode.
    """
    from webrtc import HumanPlayer
    
    # 从应用上下文中获取所需对象
    opt = app['opt']
    model = app['model']
    avatar = app['avatar']
    nerfreals = app['nerfreals']
    pcs = app['pcs']
    
    # 构建虚拟人实例
    nerfreal = await asyncio.get_event_loop().run_in_executor(
        None, build_nerfreal, opt, model, avatar, sessionid
    )
    nerfreals[sessionid] = nerfreal
    
    # 创建WebRTC对等连接
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # 设置媒体轨道
    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.audio)
    pc.addTrack(player.video)
    
    # 创建并设置本地SDP
    await pc.setLocalDescription(await pc.createOffer())
    # 推送SDP到服务器并获取应答
    answer_sdp = await post(push_url, pc.localDescription.sdp)
    # 设置远程SDP
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type='answer'))

