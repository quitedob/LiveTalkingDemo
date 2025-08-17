# pkg/tts/tencent/engine.py
"""
TencentTTS引擎实现
基于腾讯云语音合成服务的TTS引擎
"""
import os
import time
import hmac
import hashlib
import base64
import json
import uuid
import numpy as np
import requests
from typing import Iterator

from ..base import BaseTTS, State
from logger import logger


# 腾讯云TTS服务配置
_PROTOCOL = "https://"
_HOST = "tts.cloud.tencent.com"
_PATH = "/stream"
_ACTION = "TextToStreamAudio"


class TencentTTS(BaseTTS):
    """
    TencentTTS引擎类
    基于腾讯云语音合成服务的TTS引擎
    """
    
    def __init__(self, opt, parent):
        """
        初始化腾讯TTS引擎
        :param opt: 配置选项
        :param parent: BaseReal实例，用于回调
        """
        super().__init__(opt, parent)
        self.appid = os.getenv("TENCENT_APPID")
        self.secret_key = os.getenv("TENCENT_SECRET_KEY")
        self.secret_id = os.getenv("TENCENT_SECRET_ID")
        self.voice_type = int(opt.REF_FILE)
        self.codec = "pcm"
        self.sample_rate = 16000
        self.volume = 0
        self.speed = 0
    
    def __gen_signature(self, params):
        """
        生成腾讯云API签名
        :param params: 请求参数
        :return: 签名字符串
        """
        sort_dict = sorted(params.keys())
        sign_str = "POST" + _HOST + _PATH + "?"
        for key in sort_dict:
            sign_str = sign_str + key + "=" + str(params[key]) + '&'
        sign_str = sign_str[:-1]
        hmacstr = hmac.new(self.secret_key.encode('utf-8'),
                           sign_str.encode('utf-8'), hashlib.sha1).digest()
        s = base64.b64encode(hmacstr)
        s = s.decode('utf-8')
        return s

    def __gen_params(self, session_id, text):
        """
        生成请求参数
        :param session_id: 会话ID
        :param text: 要合成的文本
        :return: 请求参数字典
        """
        params = dict()
        params['Action'] = _ACTION
        params['AppId'] = int(self.appid)
        params['SecretId'] = self.secret_id
        params['ModelType'] = 1
        params['VoiceType'] = self.voice_type
        params['Codec'] = self.codec
        params['SampleRate'] = self.sample_rate
        params['Speed'] = self.speed
        params['Volume'] = self.volume
        params['SessionId'] = session_id
        params['Text'] = text

        timestamp = int(time.time())
        params['Timestamp'] = timestamp
        params['Expired'] = timestamp + 24 * 60 * 60
        return params

    def txt_to_audio(self, msg):
        """
        将文本转换为音频
        :param msg: 包含文本、事件和选项的元组
        """
        # 解包三元组
        text, textevent, tts_options = msg
        self.stream_tts(
            self.tencent_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh",  # en args.language,
                self.opt.TTS_SERVER,  # "http://127.0.0.1:5000", #args.server_url,
            ),
            msg  # 传递完整元组
        )

    def tencent_voice(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        """
        调用腾讯云TTS服务进行语音合成
        :param text: 要合成的文本
        :param reffile: 参考音频文件路径
        :param reftext: 参考音频对应的文本
        :param language: 语言代码
        :param server_url: 服务器地址
        :return: 音频字节流迭代器
        """
        start = time.perf_counter()
        session_id = str(uuid.uuid1())
        params = self.__gen_params(session_id, text)
        signature = self.__gen_signature(params)
        headers = {
            "Content-Type": "application/json",
            "Authorization": str(signature)
        }
        url = _PROTOCOL + _HOST + _PATH
        try:
            res = requests.post(url, headers=headers,
                              data=json.dumps(params), stream=True)
            
            end = time.perf_counter()
            logger.info(f"tencent Time to make POST: {end-start}s")
                
            first = True
        
            for chunk in res.iter_content(chunk_size=6400):  # 640 16K*20ms*2
                if first:
                    try:
                        rsp = json.loads(chunk)
                        logger.error("tencent tts:%s", rsp["Response"]["Error"]["Message"])
                        return
                    except:
                        end = time.perf_counter()
                        logger.info(f"tencent Time to first chunk: {end-start}s")
                        first = False                    
                if chunk and self.state == State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('tencent')

    def stream_tts(self, audio_stream, msg):
        """
        处理流式TTS音频数据
        :param audio_stream: 音频流迭代器
        :param msg: 包含文本、事件和选项的元组
        """
        # 解包三元组
        text, textevent, tts_options = msg
        first = True
        last_stream = np.array([], dtype=np.float32)
        for chunk in audio_stream:
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
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:]  # get the remain stream
        eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
