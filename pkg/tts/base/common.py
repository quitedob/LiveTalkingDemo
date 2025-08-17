# pkg/tts/base/common.py
"""
TTS基础类和公共工具模块
包含所有TTS引擎的基础类、状态枚举和工具函数
"""
from __future__ import annotations
import re
import numpy as np
import soundfile as sf
import resampy
import queue
from queue import Queue
from io import BytesIO
from threading import Thread
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from basereal import BaseReal

from logger import logger


class State(Enum):
    """TTS状态枚举"""
    RUNNING = 0  # 运行状态
    PAUSE = 1    # 暂停状态


def _sentence_splitter(text: str):
    """
    使用正则表达式按标点符号分割文本成句子。
    Splits text into sentences using regular expressions based on punctuation.
    """
    # 使用正则表达式按常见的结束标点分割句子
    sentences = re.split(r'([。？！；!?;\n])', text)
    # 将标点符号重新附加到句子末尾
    result = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
              for i in range(0, len(sentences), 2)]
    # 过滤掉可能产生的空字符串
    return [s.strip() for s in result if s.strip()]


class BaseTTS:
    """
    TTS基础类，定义了所有TTS引擎的通用接口和基础功能
    """
    def __init__(self, opt, parent: BaseReal):
        """
        初始化TTS基础类
        :param opt: 配置选项
        :param parent: BaseReal实例，用于回调
        """
        self.opt = opt
        self.parent = parent

        self.fps = opt.fps  # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps  # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        # 统一队列变量名为 msg_queue，以匹配上下文
        self.msg_queue = Queue()
        self.state = State.RUNNING

    def flush_talk(self):
        """清空消息队列并暂停TTS"""
        self.msg_queue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self, msg: str, eventpoint=None, **tts_options):
        """
        向 TTS 工作线程写入文本。
        :param msg: 文本内容
        :param eventpoint: 上游同步事件
        :param tts_options: 语速/情感/角色等可扩展参数
        """
        if len(msg) > 0:
            # 将包括 tts_options 在内的参数整体塞入队列，后续线程自行解析
            self.msg_queue.put((msg, eventpoint, tts_options))

    def render(self, quit_event):
        """启动TTS处理线程"""
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self, quit_event):
        """TTS处理主循环"""
        while not quit_event.is_set():
            try:
                # 从队列中获取包含 tts_options 的完整元组
                msg_tuple = self.msg_queue.get(block=True, timeout=1)
                self.state = State.RUNNING
            except queue.Empty:
                continue
            # 将完整的元组传递给 txt_to_audio
            self.txt_to_audio(msg_tuple)
        logger.info('ttsreal thread stop')
    
    def txt_to_audio(self, msg):
        """
        将文本转换为音频的抽象方法，需要子类实现
        :param msg: 包含文本、事件和选项的元组
        """
        pass
