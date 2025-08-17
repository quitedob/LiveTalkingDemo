# pkg/tts/xtts/engine.py
"""
XTTS引擎实现
基于XTTS的语音合成引擎
"""
import time
import numpy as np
import resampy
import requests
from typing import Iterator

from ..base import BaseTTS, State
from logger import logger


class XTTS(BaseTTS):
    """
    XTTS引擎类
    基于XTTS的语音合成引擎
    """
    
    def __init__(self, opt, parent):
        """
        初始化XTTS引擎
        :param opt: 配置选项
        :param parent: BaseReal实例，用于回调
        """
        super().__init__(opt, parent)
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self, msg):
        """
        将文本转换为音频
        :param msg: 包含文本、事件和选项的元组
        """
        # 解包三元组
        text, textevent, tts_options = msg
        self.stream_tts(
            self.xtts(
                text,
                self.speaker,
                "zh-cn",  # en args.language,
                self.opt.TTS_SERVER,  # "http://localhost:9000", #args.server_url,
                "20"  # args.stream_chunk_size
            ),
            msg  # 传递完整元组
        )

    def get_speaker(self, ref_audio, server_url):
        """
        获取说话人信息
        :param ref_audio: 参考音频文件路径
        :param server_url: 服务器地址
        :return: 说话人信息字典
        """
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()

    def xtts(self, text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        """
        调用XTTS服务进行语音合成
        :param text: 要合成的文本
        :param speaker: 说话人信息
        :param language: 语言代码
        :param server_url: 服务器地址
        :param stream_chunk_size: 流式块大小
        :return: 音频字节流迭代器
        """
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        try:
            res = requests.post(
                f"{server_url}/tts_stream",
                json=speaker,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"xtts Time to make POST: {end-start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True
        
            for chunk in res.iter_content(chunk_size=9600):  # 24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"xtts Time to first chunk: {end-start}s")
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            print(e)
    
    def stream_tts(self, audio_stream, msg):
        """
        处理流式TTS音频数据
        :param audio_stream: 音频流迭代器
        :param msg: 包含文本、事件和选项的元组
        """
        # 解包三元组
        text, textevent, tts_options = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:     
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = None
                    if first:
                        eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
