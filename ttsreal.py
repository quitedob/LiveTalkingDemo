# /workspace/LiveTalking/ttsreal.py
"""
TTS模块主入口文件
重构后的TTS系统，使用模块化的TTS引擎
"""
from __future__ import annotations

# 导入所有TTS引擎和基础组件
from pkg.tts import (
    BaseTTS, 
    State, 
    _sentence_splitter,
    EdgeTTS,
    FishTTS,
    SovitsTTS,
    CosyVoiceTTS,
    TencentTTS,
    DoubaoTTS,
    XTTS,
    TTS_ENGINES
)

# 为了保持向后兼容性，重新导出所有类
__all__ = [
    'BaseTTS',
    'State', 
    '_sentence_splitter',
    'EdgeTTS',
    'FishTTS', 
    'SovitsTTS',
    'CosyVoiceTTS',
    'TencentTTS',
    'DoubaoTTS',
    'XTTS',
    'TTS_ENGINES'
]

# 兼容性别名（如果有其他代码使用了旧的类名）
# 这些可以根据实际使用情况进行调整
