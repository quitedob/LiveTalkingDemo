# pkg/tts/fishtts/__init__.py
"""
FishTTS引擎模块
通过 HTTP/msgpack 与 FishTTS 服务交互的 TTS 引擎
支持默认音色合成和基于参考音频的零样本音色克隆
"""
from .engine import FishTTS

__all__ = ['FishTTS']
