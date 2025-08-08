# /workspace/LiveTalking/app.py
# -*- 简化注释：导入必要的库 -*-
import argparse
import asyncio
import json
import ssl
from pathlib import Path

import aiohttp_cors
import torch
import torch.multiprocessing as mp
from aiohttp import web

# 导入API路由注册函数
from api.control_api import register_control_routes
from api.human_api import register_human_routes
from api.rag_api import register_rag_routes
from api.utils import run_push_session
from api.voice_clone_api import register_voice_clone_routes
# 导入模型和客户端
from funasr import AutoModel
from llm import LLMClient
from logger import logger
from ttsreal import CosyVoiceTTS


async def on_shutdown(app):
    """
    服务器关闭时，清理所有WebRTC连接。
    When the server shuts down, clean up all WebRTC connections.
    """
    # 从应用上下文中获取 pcs 集合
    pcs = app.get('pcs', set())
    # 并发关闭所有连接
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    logger.info("所有 WebRTC 连接已成功关闭。")


def main():
    """
    主函数：解析参数、加载模型、设置并启动服务器。
    Main function: parses arguments, loads models, sets up and starts the server.
    """
    # --- 1. 参数解析 ---
    parser = argparse.ArgumentParser()
    # 添加所有命令行参数
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
    parser.add_argument('--cosyvoice-model-path', type=str, default='pretrained_models/CosyVoice2-0.5B', help="Path to the CosyVoice pretrained model directory.")
    parser.add_argument('--cloned-voices-path', type=str, default='./cloned_voices', help="Directory to store cloned voice audio and metadata.")
    
    opt = parser.parse_args()

    # --- 重新添加被遗漏的 customopt 初始化逻辑 ---
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r', encoding='utf-8') as file:
            opt.customopt = json.load(file)
    else:
        opt.customopt = []


    # --- 2. aiohttp 应用初始化 ---
    app = web.Application(client_max_size=1024**2*10) # 设置最大请求体为10MB
    app.on_shutdown.append(on_shutdown)
    
    # 将所有运行时对象存储在app上下文中，方便在请求处理器中访问
    app['opt'] = opt
    app['nerfreals'] = {}
    app['pcs'] = set()
    app['model'] = None
    app['avatar'] = None
    
    # --- 3. 加载核心服务和模型 ---
    # 加载LLM客户端
    app['llm_client'] = LLMClient(url=opt.ollama_url, model=opt.ollama_model, system_prompt=opt.ollama_system_prompt)
    logger.info("LLM 客户端已初始化。")

    # 加载主模型 (MuseTalk, Wav2Lip, etc.)
    if opt.model == 'musetalk':
        from musereal import load_model, load_avatar, warm_up
        app['model'] = load_model()
        app['avatar'] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, app['model'])
    elif opt.model == 'wav2lip':
        from lipreal import load_model, load_avatar, warm_up
        app['model'] = load_model("./models/wav2lip.pth")
        app['avatar'] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, app['model'], 256)
    elif opt.model == 'ultralight':
        from lightreal import load_model, load_avatar, warm_up
        app['model'] = load_model(opt)
        app['avatar'] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, app['avatar'], 160)
    logger.info(f"主模型 '{opt.model}' 加载完成。")

    # 加载ASR模型
    logger.info('正在加载 FunASR 模型...')
    try:
        app['asr_model'] = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True, vad_model="fsmn-vad", disable_update=True, vad_kwargs={"max_single_segment_time": 30000}, device="cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info('FunASR 模型加载成功。')
    except Exception as e:
        logger.error(f'加载 FunASR 模型失败: {e}')
        app['asr_model'] = None

    # 初始化CosyVoice TTS实例 (如果配置需要)
    app['cosyvoice_tts_instance'] = None
    if opt.tts == 'cosyvoice':
        try:
            logger.info("正在初始化全局 CosyVoiceTTS 实例...")
            app['cosyvoice_tts_instance'] = CosyVoiceTTS(opt)
            logger.info("全局 CosyVoiceTTS 实例初始化完成。")
        except Exception as e:
            logger.error(f"初始化全局 CosyVoiceTTS 实例失败: {e}")
            
    # --- 4. 注册所有API路由 ---
    register_control_routes(app)
    register_human_routes(app)
    register_voice_clone_routes(app)
    register_rag_routes(app)
    app.router.add_static('/', path='web') # 提供静态文件服务
    logger.info("所有API路由注册完成。")

    # 配置CORS
    cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")})
    for route in list(app.router.routes()):
        cors.add(route)
    logger.info("CORS策略已应用到所有路由。")
    
    # --- 5. 启动服务器 ---
    runner = web.AppRunner(app)
    
    async def start_server():
        await runner.setup()
        # 配置SSL上下文
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        try:
            ssl_context.load_cert_chain(opt.cert_path, opt.key_path)
            logger.info(f"成功加载SSL证书: {opt.cert_path}")
        except FileNotFoundError:
            logger.error(f"SSL证书或密钥文件未找到: {opt.cert_path} 或 {opt.key_path}。服务器将以HTTP模式启动。")
            ssl_context = None
        except ssl.SSLError as e:
            logger.error(f"加载SSL证书时发生错误: {e}。服务器将以HTTP模式启动。")
            ssl_context = None
        
        # 启动TCP站点
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport, ssl_context=ssl_context)
        await site.start()
        logger.info(f'服务器已在 0.0.0.0:{opt.listenport} 上启动 ({"HTTPS" if ssl_context else "HTTP"})')

        # 如果是 rtcpush 模式，启动推流会话
        if opt.transport == 'rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url + (str(k) if k != 0 else "")
                asyncio.create_task(run_push_session(app, push_url, k))
                logger.info(f"已为会话 {k} 启动 RTC 推流任务。")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server())
    
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("检测到键盘中断，正在关闭服务器...")
    finally:
        loop.run_until_complete(runner.cleanup())
        logger.info("服务器已成功关闭。")

if __name__ == '__main__':
    # 设置多进程启动方法为 'spawn'，以避免CUDA初始化问题
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # 确保声音克隆目录存在
    CLONED_VOICES_PATH = Path('./cloned_voices')
    CLONED_VOICES_PATH.mkdir(exist_ok=True)
    (CLONED_VOICES_PATH / 'audio').mkdir(exist_ok=True)
    
    main()
