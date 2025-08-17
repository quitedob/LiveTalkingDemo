# pkg/tts/base/__init__.py
"""
TTS基础模块
导出基础类和公共工具
"""
from .common import BaseTTS, State, _sentence_splitter

__all__ = ['BaseTTS', 'State', '_sentence_splitter']
