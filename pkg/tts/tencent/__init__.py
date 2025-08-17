# pkg/tts/tencent/__init__.py
"""
TencentTTS引擎模块
基于腾讯云语音合成服务的TTS引擎
"""
from .engine import TencentTTS

__all__ = ['TencentTTS']
