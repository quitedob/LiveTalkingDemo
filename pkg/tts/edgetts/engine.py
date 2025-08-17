# pkg/tts/edgetts/engine.py
"""
EdgeTTS引擎实现
基于微软Edge浏览器的文本转语音服务
"""
import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts
from io import BytesIO

from ..base import BaseTTS, State, _sentence_splitter
from logger import logger


class EdgeTTS(BaseTTS):
    """
    EdgeTTS引擎类
    使用微软Edge浏览器的TTS服务进行语音合成
    """
    
    def txt_to_audio(self, msg):
        """
        将文本转换为音频
        :param msg: 包含文本、事件和选项的元组
        """
        voicename = self.opt.REF_FILE  # "zh-CN-YunxiaNeural"
        text, textevent, tts_options = msg

        # 将长文本分割成句子
        sentences = _sentence_splitter(text)
        
        # 标记第一个和最后一个句子，用于发送 'start' 和 'end' 事件
        total_sentences = len(sentences)
        for i, sentence in enumerate(sentences):
            if not self.state == State.RUNNING:
                logger.warning("TTS 任务在处理句子时被中断。")
                break
            
            # 清空上一句的音频流
            self.input_stream.seek(0)
            self.input_stream.truncate()

            t = time.time()
            # 逐句进行TTS合成
            asyncio.run(self.__main(voicename, sentence))
            logger.info(f'-------edge tts time for sentence: {time.time()-t:.4f}s')

            if self.input_stream.getbuffer().nbytes <= 0:
                logger.error(f'EdgeTTS 合成句子失败: "{sentence}"')
                continue
            
            self.input_stream.seek(0)
            stream = self.__create_bytes_stream(self.input_stream)
            streamlen = stream.shape[0]
            idx = 0

            is_first_chunk_of_all = (i == 0)
            
            chunk_index = 0
            while streamlen >= self.chunk and self.state == State.RUNNING:
                eventpoint = None
                
                # 只有在整个文本的第一个句子的第一个音频块，才发送 'start' 事件
                if is_first_chunk_of_all and chunk_index == 0:
                    eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}

                self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                
                streamlen -= self.chunk
                idx += self.chunk
                chunk_index += 1

        # 所有句子处理完毕后，发送 'end' 事件
        if self.state == State.RUNNING:
            end_event = {'status': 'end', 'text': text, 'msgevent': textevent}
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), end_event)

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
    
    async def __main(self, voicename: str, text: str):
        """
        EdgeTTS异步合成方法
        :param voicename: 语音名称
        :param text: 要合成的文本
        """
        try:
            communicate = edge_tts.Communicate(text, voicename)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and self.state == State.RUNNING:
                    self.input_stream.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logger.exception('edgetts')
