# pkg/tts/doubao/engine.py
"""
DoubaoTTS引擎实现
基于抖音豆包TTS服务的语音合成引擎
"""
import os
import time
import json
import uuid
import copy
import gzip
import asyncio
import websockets
import numpy as np

from ..base import BaseTTS, State
from logger import logger


class DoubaoTTS(BaseTTS):
    """
    DoubaoTTS引擎类
    基于抖音豆包TTS服务的语音合成引擎
    """
    
    def __init__(self, opt, parent):
        """
        初始化抖音豆包TTS客户端；自动根据 REF_FILE 决定集群（S_ 前缀→volcano_icl）
        :param opt: 配置选项
        :param parent: BaseReal实例，用于回调
        """
        super().__init__(opt, parent)
        # 读取凭证（需提前 export DOUBAO_APPID / DOUBAO_TOKEN）
        self.appid = os.getenv("DOUBAO_APPID")
        self.token = os.getenv("DOUBAO_TOKEN")

        # 关键修复：允许通过环境变量覆盖集群；若 REF_FILE 以 "S_" 开头（克隆声线），默认用 volcano_icl
        ref = (opt.REF_FILE or "").strip()
        env_cluster = os.getenv("DOUBAO_CLUSTER", "").strip()
        if env_cluster:
            _cluster = env_cluster
        else:
            _cluster = "volcano_icl" if ref.startswith("S_") else "volcano_tts"

        # WebSocket 地址保持不变
        _host = "openspeech.bytedance.com"
        self.api_url = f"wss://{_host}/api/v1/tts/ws_binary"

        # 组装默认请求体（后面会在 doubao_voice 里覆写 voice_type / text / reqid）
        self.request_json = {
            "app": {
                "appid": self.appid,
                # 注意：这里的 token 字段不用填 access_token；真正的鉴权走 WS 头部 Authorization
                "token": "access_token",
                "cluster": _cluster,  # 这里使用上面推断/外部传入的集群
            },
            "user": { "uid": "xxx" },
            "audio": {
                "voice_type": "xxx",
                # 项目里按 PCM int16 解码流，所以保持 pcm 编码
                "encoding": "pcm",
                "rate": 16000,
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": "xxx",
                "text": "字节跳动语音合成。",
                "text_type": "plain",
                "operation": "xxx",
            },
        }

        # 可选健壮性日志：帮助定位未设置 token 的情况（避免出现 'Bearer;'）
        if not self.token:
            logger.warning("DoubaoTTS: DOUBAO_TOKEN 未设置，WS 将以 'Bearer;' 发送，可能导致无音频返回。")
            
    def put_msg_txt(self, msg, eventpoint=None, **tts_options):
        """
        覆盖父类方法，先把可映射的参数写入 self.request_json，再走父类逻辑
        :param msg: 文本内容
        :param eventpoint: 上游同步事件
        :param tts_options: 语速/情感/角色等可扩展参数
        """
        # 把 tts_options 映射进 request_json
        audio_cfg = self.request_json["audio"]
        if "speed" in tts_options:
            audio_cfg["speed_ratio"] = float(tts_options["speed"])
        if "volume" in tts_options:
            audio_cfg["volume_ratio"] = float(tts_options["volume"])
        if "pitch" in tts_options:
            audio_cfg["pitch_ratio"] = float(tts_options["pitch"])
        if "emotion" in tts_options:
            audio_cfg["enable_emotion"] = True
            audio_cfg["emotion"] = tts_options["emotion"]
        
        # 调用父类方法，将消息放入队列，保持工作流一致
        super().put_msg_txt(msg, eventpoint, **tts_options)

    async def doubao_voice(self, text):
        """
        调用豆包TTS服务进行语音合成
        :param text: 要合成的文本
        :return: 音频字节流异步迭代器
        """
        start = time.perf_counter()
        voice_type = self.opt.REF_FILE

        try:
            # 创建请求对象
            default_header = bytearray(b'\x11\x10\x11\x00')
            # 使用已经根据 tts_options 修改过的 request_json
            submit_request_json = copy.deepcopy(self.request_json)
            submit_request_json["user"]["uid"] = self.parent.sessionid
            submit_request_json["audio"]["voice_type"] = voice_type
            submit_request_json["request"]["text"] = text
            submit_request_json["request"]["reqid"] = str(uuid.uuid4())
            submit_request_json["request"]["operation"] = "submit"
            payload_bytes = str.encode(json.dumps(submit_request_json))
            payload_bytes = gzip.compress(payload_bytes)  # if no compression, comment this line
            full_client_request = bytearray(default_header)
            full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
            full_client_request.extend(payload_bytes)  # payload

            header = {"Authorization": f"Bearer; {self.token}"}
            first = True
            async with websockets.connect(self.api_url, extra_headers=header, ping_interval=None) as ws:
                await ws.send(full_client_request)
                while True:
                    res = await ws.recv()
                    header_size = res[0] & 0x0f
                    message_type = res[1] >> 4
                    message_type_specific_flags = res[1] & 0x0f
                    payload = res[header_size*4:]

                    if message_type == 0xb:  # audio-only server response
                        if message_type_specific_flags == 0:  # no sequence number as ACK
                            continue
                        else:
                            if first:
                                end = time.perf_counter()
                                logger.info(f"doubao tts Time to first chunk: {end-start}s")
                                first = False
                            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                            payload = payload[8:]
                            yield payload
                        if sequence_number < 0:
                            break
                    else:
                        break
        except Exception as e:
            logger.exception('doubao')

    def txt_to_audio(self, msg):
        """
        将文本转换为音频
        :param msg: 包含文本、事件和选项的元组
        """
        # 解包三元组
        text, textevent, tts_options = msg
        asyncio.new_event_loop().run_until_complete(
            self.stream_tts(
                self.doubao_voice(text),
                msg  # 传递完整元组
            )
        )

    async def stream_tts(self, audio_stream, msg):
        """
        处理流式TTS音频数据
        :param audio_stream: 音频流异步迭代器
        :param msg: 包含文本、事件和选项的元组
        """
        # 解包三元组
        text, textevent, tts_options = msg
        first = True
        last_stream = np.array([], dtype=np.float32)
        async for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = np.concatenate((last_stream, stream))
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = None
                    if first:
                        eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:]  # get the remain stream
        
        eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
