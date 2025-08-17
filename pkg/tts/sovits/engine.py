# pkg/tts/sovits/engine.py
"""
SovitsTTS引擎实现
基于GPT-SoVITS的语音合成引擎
"""
import time
import numpy as np
import soundfile as sf
import resampy
import requests
from io import BytesIO
from typing import Iterator

from ..base import BaseTTS, State
from logger import logger


class SovitsTTS(BaseTTS):
    """
    SovitsTTS引擎类
    基于GPT-SoVITS的语音合成引擎
    """
    
    def txt_to_audio(self, msg): 
        """
        将文本转换为音频
        :param msg: 包含文本、事件和选项的元组
        """
        # 解包三元组
        text, textevent, tts_options = msg
        self.stream_tts(
            self.gpt_sovits(
                text=text,
                reffile=self.opt.REF_FILE,
                reftext=self.opt.REF_TEXT,
                language="zh",  # en args.language,
                server_url=self.opt.TTS_SERVER,  # "http://127.0.0.1:5000", #args.server_url,
            ),
            msg  # 传递完整元组
        )

    def gpt_sovits(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        """
        调用GPT-SoVITS服务进行语音合成
        :param text: 要合成的文本
        :param reffile: 参考音频文件路径
        :param reftext: 参考音频对应的文本
        :param language: 语言代码
        :param server_url: 服务器地址
        :return: 音频字节流迭代器
        """
        start = time.perf_counter()
        req = {
            'text': text,
            'text_lang': language,
            'ref_audio_path': reffile,
            'prompt_text': reftext,
            'prompt_lang': language,
            'media_type': 'ogg',
            'streaming_mode': True
        }
        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=None):  # 12800 1280 32K*20ms*2
                logger.info('chunk len:%d', len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('sovits')

    def __create_bytes_stream(self, byte_stream):
        """
        创建音频字节流
        :param byte_stream: 字节流对象
        :return: 处理后的音频数组
        """
        stream, sample_rate = sf.read(byte_stream)  # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

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
                byte_stream = BytesIO(chunk)
                stream = self.__create_bytes_stream(byte_stream)
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
