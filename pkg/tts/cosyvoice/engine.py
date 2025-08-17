# pkg/tts/cosyvoice/engine.py
"""
CosyVoiceTTS引擎实现
基于CosyVoice的语音合成引擎
"""
import time
import numpy as np
import resampy
import requests
from typing import Iterator

from ..base import BaseTTS, State
from logger import logger


class CosyVoiceTTS(BaseTTS):
    """
    CosyVoiceTTS引擎类
    基于CosyVoice的语音合成引擎
    """
    
    def txt_to_audio(self, msg):
        """
        将文本转换为音频
        :param msg: 包含文本、事件和选项的元组
        """
        # 解包三元组
        text, textevent, tts_options = msg
        self.stream_tts(
            self.cosy_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh",  # en args.language,
                self.opt.TTS_SERVER,  # "http://127.0.0.1:5000", #args.server_url,
            ),
            msg  # 传递完整元组
        )

    def cosy_voice(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        """
        调用CosyVoice服务进行语音合成
        :param text: 要合成的文本
        :param reffile: 参考音频文件路径
        :param reftext: 参考音频对应的文本
        :param language: 语言代码
        :param server_url: 服务器地址
        :return: 音频字节流迭代器
        """
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        try:
            files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]
            res = requests.request("GET", f"{server_url}/inference_zero_shot", 
                                 data=payload, files=files, stream=True)
            
            end = time.perf_counter()
            logger.info(f"cosy_voice Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=9600):  # 960 24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"cosy_voice Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('cosyvoice')

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
