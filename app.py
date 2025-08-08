# /workspace/LiveTalking/app.py

# -*- 简化注释：导入必要的库 -*-
import ssl
import json
import uuid
import asyncio
import argparse
import random
import os
import gc
import tempfile
from functools import partial
from pathlib import Path
from threading import Thread, Event
import torch.multiprocessing as mp
import time
from typing import Dict

# aiohttp 和 webrtc 相关导入
from aiohttp import web
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

# 项目内部模块导入
from logger import logger
from basereal import BaseReal
from webrtc import HumanPlayer
from ttsreal import CosyVoiceTTS
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from llm import LLMClient
import torch
import numpy as np


# --- 全局变量定义 ---
nerfreals: Dict[int, BaseReal] = {}
opt = None
model = None
avatar = None
asr_model = None
llm_client = None
pcs = set()
cosyvoice_tts_instance = None # 全局 CosyVoiceTTS 实例，用于声音克隆

# --- 确保克隆声音的存储目录存在 ---
CLONED_VOICES_PATH = Path('./cloned_voices')
CLONED_VOICES_PATH.mkdir(exist_ok=True)
(CLONED_VOICES_PATH / 'audio').mkdir(exist_ok=True)


def randN(N) -> int:
    """生成一个指定长度的随机整数"""
    min_val = pow(10, N - 1)
    max_val = pow(10, N)
    return random.randint(min_val, max_val - 1)


def build_nerfreal(sessionid: int) -> BaseReal:
    """根据配置构建并返回一个虚拟人实例"""
    opt.sessionid = sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt, model, avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt, model, avatar)
    elif opt.model == 'ernerf':
        pass # ernerf 模型暂未实现
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt, model, avatar)
    return nerfreal


# --- 新增API端点 ---

async def get_cloned_voices(request):
    """获取所有已克隆的声音列表"""
    if not cosyvoice_tts_instance:
        return web.Response(
            content_type="application/json", status=503,
            text=json.dumps({"error": "CosyVoice service not available"})
        )
    try:
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
    """处理音频上传，克隆新声音"""
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

        new_voice_profile = await asyncio.get_event_loop().run_in_executor(
            None,
            cosyvoice_tts_instance.add_cloned_voice,
            voice_name,
            audio_bytes
        )
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

# --- 现有API端点（部分已修改） ---

async def offer(request):
    """处理WebRTC的offer请求，建立P2P连接"""
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    sessionid = randN(6)
    nerfreals[sessionid] = None
    logger.info('创建会话 sessionid=%d, 当前会话数=%d', sessionid, len(nerfreals))

    ice_servers = [RTCIceServer(urls="stun:stun.miwifi.com:3478")]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("连接状态变为: %s", pc.connectionState)
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            if sessionid in nerfreals:
                logger.info(f"因连接状态为 {pc.connectionState}，正在关闭会话 {sessionid}")
                session_to_close = nerfreals.pop(sessionid, None)
                if session_to_close:
                    del session_to_close
                gc.collect()
            if pc in pcs:
                await pc.close()
                pcs.discard(pc)

    logger.info(f"会话 {sessionid}: 开始构建虚拟人实例...")
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    logger.info(f"会话 {sessionid}: 虚拟人实例构建完成。")

    if sessionid not in nerfreals:
        logger.warning(f"会话 {sessionid} 在实例准备好之前已关闭。")
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"code": -1, "msg": "连接在设置过程中失败"})
        )
    nerfreals[sessionid] = nerfreal

    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.audio)
    pc.addTrack(player.video)

    # ==================== 代码修复位置 S T A R T ====================
    # 统一处理 aiortc 不同版本返回的 capabilities 格式，修复 AttributeError
    try:
        capabilities = RTCRtpSender.getCapabilities("video")  # 获取本机支持的 video capabilities
        
        # 兼容 dataclass (有 .codecs 属性) 与 dict (有 ['codecs'] 键)
        codecs_list = getattr(capabilities, "codecs", None) or capabilities.get("codecs", [])
        
        if codecs_list:
            # 仅保留 H264/VP8 两种常见且浏览器普遍支持的编码器
            preferences = [c for c in codecs_list if c.mimeType.upper().endswith(("H264", "VP8"))]
            video_transceiver = next(
                (t for t in pc.getTransceivers()
                 if t.sender and t.sender.track and t.sender.track.kind == "video"),
                None
            )
            if video_transceiver and preferences:
                video_transceiver.setCodecPreferences(preferences)
                logger.info("已成功设置视频首选编码器为: %s", [c.mimeType for c in preferences])
    except Exception as e:
        logger.warning("设置编解码器偏好失败: %s", e)
    # ==================== 代码修复位置 E N D ======================

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid})
    )


async def human(request):
    """处理文本输入，驱动虚拟人进行交谈或复述"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        if not sessionid or sessionid not in nerfreals:
            raise ValueError("无效或已过期的 sessionid")

        if params.get('interrupt'):
            nerfreals[sessionid].flush_talk()

        # 获取 TTS 选项
        tts_options = params.get('tts_options', {})

        if params['type'] == 'echo':
            # 将 tts_options 传递下去
            nerfreals[sessionid].put_msg_txt(params['text'], tts_options=tts_options)
        elif params['type'] == 'chat':
            async def stream_llm_to_tts():
                try:
                    response_chunks = []
                    async for text_chunk in llm_client.ask(params['text']):
                        if sessionid not in nerfreals:
                            logger.warning(f"会话 {sessionid} 在接收LLM响应时已关闭，中断任务。")
                            return
                        response_chunks.append(text_chunk)
                    
                    full_response = "".join(response_chunks)

                    if full_response and sessionid in nerfreals:
                        # 将 tts_options 传递下去
                        nerfreals[sessionid].put_msg_txt(full_response, tts_options=tts_options)

                except Exception as e:
                    logger.error(f"stream_llm_to_tts 任务执行出错 (会话 {sessionid}): {e}")

            asyncio.create_task(stream_llm_to_tts())

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


async def interrupt_talk(request):
    """中断虚拟人当前的语音播报"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        if sessionid in nerfreals:
            nerfreals[sessionid].flush_talk()
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
    except Exception as e:
        logger.exception('interrupt_talk接口异常:')
        return web.Response(content_type="application/json", status=500, text=json.dumps({"code": -1, "msg": str(e)}))


async def humanaudio(request):
    """处理上传的音频文件，驱动虚拟人"""
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        if sessionid in nerfreals:
            fileobj = form["file"]
            filebytes = fileobj.file.read()
            nerfreals[sessionid].put_audio_file(filebytes)
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
    except Exception as e:
        logger.exception('humanaudio接口异常:')
        return web.Response(content_type="application/json", status=500, text=json.dumps({"code": -1, "msg": str(e)}))


async def set_audiotype(request):
    """设置音频类型或重新初始化虚拟人状态"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        if sessionid in nerfreals:
            nerfreals[sessionid].set_custom_state(params['audiotype'], params['reinit'])
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
    except Exception as e:
        logger.exception('set_audiotype接口异常:')
        return web.Response(content_type="application/json", status=500, text=json.dumps({"code": -1, "msg": str(e)}))


async def record(request):
    """控制录制功能的开始和结束"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        if sessionid in nerfreals:
            if params['type'] == 'start_record':
                nerfreals[sessionid].start_recording()
            elif params['type'] == 'end_record':
                nerfreals[sessionid].stop_recording()
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
    except Exception as e:
        logger.exception('record接口异常:')
        return web.Response(content_type="application/json", status=500, text=json.dumps({"code": -1, "msg": str(e)}))


async def is_speaking(request):
    """查询虚拟人当前是否正在讲话"""
    params = await request.json()
    sessionid = int(params.get('sessionid', 0))
    speaking = nerfreals[sessionid].is_speaking() if sessionid in nerfreals else False
    return web.Response(content_type="application/json", text=json.dumps({"code": 0, "data": speaking}))


async def transcribe_audio(audio_bytes: bytes) -> str:
    """将音频字节流通过FunASR转录为文本"""
    loop = asyncio.get_event_loop()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        generate_fn = partial(asr_model.generate, input=tmp_path, cache={}, language="zn+en", use_itn=True, batch_size_s=60, merge_vad=True, merge_length_s=15)
        res = await loop.run_in_executor(None, generate_fn)
        res = res[0] if isinstance(res, list) and res else res if isinstance(res, dict) else {}
        text_raw = res.get("text", "")
        if text_raw:
            processed_text = rich_transcription_postprocess(text_raw)
            logger.info(f"ASR 转写结果: {processed_text}")
            return processed_text
        return ""
    except Exception as e:
        logger.error(f"ASR转写出错: {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


async def audio_chat(request):
    """处理完整的语音聊天流程（ASR -> LLM -> TTS）"""
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        if not sessionid or sessionid not in nerfreals:
            return web.Response(content_type="application/json", status=400, text=json.dumps({"code": -1, "msg": "需要有效的sessionid"}))

        fileobj = form.get("file")
        if not fileobj:
            return web.Response(content_type="application/json", status=400, text=json.dumps({"code": -1, "msg": "需要音频文件"}))

        audio_bytes = fileobj.file.read()
        transcribed_text = await transcribe_audio(audio_bytes)
        if not transcribed_text:
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ASR未能识别出文本。"}))

        # 同样为语音聊天模式传递TTS选项。
        # 注意：前端需要将tts_options作为表单中的一个JSON字符串字段来传递。
        tts_options_str = form.get('tts_options', '{}')
        tts_options = json.loads(tts_options_str)

        async def stream_llm_to_tts():
            try:
                response_chunks = []
                async for text_chunk in llm_client.ask(transcribed_text):
                    if sessionid not in nerfreals:
                        logger.warning(f"会话 {sessionid} 在接收LLM响应时已关闭，中断任务。")
                        return
                    response_chunks.append(text_chunk)

                full_response = "".join(response_chunks)
                if full_response and sessionid in nerfreals:
                    nerfreals[sessionid].put_msg_txt(full_response, tts_options=tts_options)
            
            except Exception as e:
                logger.error(f"audio_chat->stream_llm_to_tts 任务出错 (会话 {sessionid}): {e}")

        asyncio.create_task(stream_llm_to_tts())

        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
    except Exception as e:
        logger.exception('audio_chat接口异常:')
        return web.Response(content_type="application/json", status=500, text=json.dumps({"code": -1, "msg": str(e)}))


async def on_shutdown(app):
    """服务器关闭时，清理所有WebRTC连接"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def post(url, data):
    """一个简单的异步POST请求函数"""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, timeout=30) as response:
                return await response.text()
    except Exception as e:
        logger.info(f'POST请求错误: {e}')


async def run_push_session(push_url, sessionid):
    """为rtcpush模式运行一个独立的虚拟人会话"""
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal
    pc = RTCPeerConnection()
    pcs.add(pc)
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.audio)
    pc.addTrack(player.video)
    await pc.setLocalDescription(await pc.createOffer())
    answer_sdp = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type='answer'))


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    # --- 现有参数 ---
    parser.add_argument('--fps', type=int, default=50, help="视频帧率")
    parser.add_argument('-l', type=int, default=10, help="滑动窗口左侧长度 (单位: 20ms)")
    parser.add_argument('-m', type=int, default=8, help="滑动窗口中间长度 (单位: 20ms)")
    parser.add_argument('-r', type=int, default=10, help="滑动窗口右侧长度 (单位: 20ms)")
    parser.add_argument('--W', type=int, default=450, help="GUI 宽度")
    parser.add_argument('--H', type=int, default=450, help="GUI 高度")
    parser.add_argument('--batch_size', type=int, default=8, help="推理批次大小, MuseTalk建议为1")
    parser.add_argument('--customvideo_config', type=str, default='', help="自定义动作json配置文件")
    parser.add_argument('--tts', type=str, default='edgetts', help="TTS服务类型 (e.g., edgetts, cosyvoice, xtts)")
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural", help="TTS参考音频或说话人")
    parser.add_argument('--REF_TEXT', type=str, default=None, help="TTS参考文本")
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880', help="TTS服务地址")
    parser.add_argument('--model', type=str, default='musetalk', help="使用的模型 (musetalk, wav2lip, ultralight)")
    parser.add_argument('--avatar_id', type=str, default='avator_1', help="定义 data/avatars 中的形象ID")
    parser.add_argument('--transport', type=str, default='webrtc', help="传输模式 (webrtc, rtcpush, virtualcam)")
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream', help="rtcpush模式下的推流地址")
    parser.add_argument('--max_session', type=int, default=1, help="最大会话数")
    parser.add_argument('--listenport', type=int, default=8010, help="Web服务监听端口")
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434/api/chat', help="Ollama聊天API的URL")
    parser.add_argument('--ollama-model', type=str, default='gemma3:12b', help="在Ollama中使用的模型名称")
    parser.add_argument('--ollama-system-prompt', type=str, default='你的身份是芝麻编程老师请你按照你的身份说话，禁止输出表情符号。/nothink', help="给Ollama模型的系统提示")
    parser.add_argument('--cert-path', default='/workspace/ssh/i.zmbc100.com_bundle.crt', help="SSL证书链文件的路径")
    parser.add_argument('--key-path', default='/workspace/ssh/i.zmbc100.com.key.noenc.pem', help="SSL私钥文件的路径")

    # --- 新增 CosyVoice 相关参数 ---
    parser.add_argument('--cosyvoice-model-path', type=str, default='pretrained_models/CosyVoice2-0.5B',
                        help="Path to the CosyVoice pretrained model directory.")
    parser.add_argument('--cloned-voices-path', type=str, default='./cloned_voices',
                        help="Directory to store cloned voice audio and metadata.")
    
    opt = parser.parse_args()

    llm_client = LLMClient(url=opt.ollama_url, model=opt.ollama_model, system_prompt=opt.ollama_system_prompt)

    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r', encoding='utf-8') as file:
            opt.customopt = json.load(file)
    else:
        opt.customopt = []

    # --- 模型加载 ---
    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        model = load_model()
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model)
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    logger.info('正在加载 FunASR 模型...')
    try:
        asr_model = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True, vad_model="fsmn-vad", disable_update=True, vad_kwargs={"max_single_segment_time": 30000}, device="cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info('FunASR 模型加载成功。')
    except Exception as e:
        logger.error(f'加载 FunASR 模型失败: {e}')

    # --- 初始化全局 CosyVoiceTTS 实例 ---
    if opt.tts == 'cosyvoice':
        try:
            logger.info("Initializing global CosyVoiceTTS instance for cloning...")
            cosyvoice_tts_instance = CosyVoiceTTS(opt)
            logger.info("Global CosyVoiceTTS instance for cloning initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize global CosyVoiceTTS instance: {e}")
            cosyvoice_tts_instance = None
            
    if opt.transport == 'virtualcam':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render, args=(thread_quit,))
        rendthrd.start()

    # --- aiohttp 应用设置和路由注册 ---
    appasync = web.Application(client_max_size=1024**2*100)
    appasync.on_shutdown.append(on_shutdown)
    
    # 注册所有路由
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/interrupt_talk", interrupt_talk)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_post("/audio_chat", audio_chat)
    appasync.router.add_get("/get_cloned_voices", get_cloned_voices)
    appasync.router.add_post("/clone_voice", clone_voice)
    appasync.router.add_static('/', path='web')

    cors = aiohttp_cors.setup(appasync, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")})
    for route in list(appasync.router.routes()):
        cors.add(route)

    # --- 服务器启动逻辑 ---
    pagename = 'webrtcapi.html'
    if opt.transport == 'rtmp': pagename = 'echoapi.html'
    elif opt.transport == 'rtcpush': pagename = 'rtcpushapi.html'

    logger.info(f'HTTPS 服务器已启动; https://<serverip>:{opt.listenport}/{pagename}')
    logger.info(f'推荐访问WebRTC集成前端: https://<serverip>:{opt.listenport}/dashboard.html')

    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        try:
            ssl_context.load_cert_chain(opt.cert_path, opt.key_path)
            logger.info(f"成功加载SSL证书: {opt.cert_path}")
        except FileNotFoundError:
            logger.error(f"SSL证书或密钥文件未找到: {opt.cert_path} 或 {opt.key_path}")
            logger.error("请确保证书文件已放置在正确目录。服务器将以HTTP模式启动。")
            ssl_context = None
        except ssl.SSLError as e:
            logger.error(f"加载SSL证书时发生错误: {e}")
            logger.error("请检查证书与私钥是否匹配。服务器将以HTTP模式启动。")
            ssl_context = None

        site = web.TCPSite(runner, '0.0.0.0', opt.listenport, ssl_context=ssl_context)
        loop.run_until_complete(site.start())

        if opt.transport == 'rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url + (str(k) if k != 0 else "")
                loop.run_until_complete(run_push_session(push_url, k))

        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("正在关闭服务器...")
    
    runner = web.AppRunner(appasync)
    run_server(runner)