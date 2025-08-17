# pkg/tts/__init__.py
"""
TTS引擎包
包含所有支持的TTS引擎实现
"""

# 导入基础类和公共工具
from .base import BaseTTS, State, _sentence_splitter

# 导入所有TTS引擎
from .edgetts import EdgeTTS
from .fishtts import FishTTS
from .sovits import SovitsTTS
from .cosyvoice import CosyVoiceTTS
from .tencent import TencentTTS
from .doubao import DoubaoTTS
from .xtts import XTTS

# 定义可导出的类和函数
__all__ = [
    # 基础类
    'BaseTTS', 
    'State', 
    '_sentence_splitter',
    
    # TTS引擎
    'EdgeTTS',
    'FishTTS', 
    'SovitsTTS',
    'CosyVoiceTTS',
    'TencentTTS',
    'DoubaoTTS',
    'XTTS'
]

# TTS引擎映射字典，便于动态创建
TTS_ENGINES = {
    'EdgeTTS': EdgeTTS,
    'FishTTS': FishTTS,
    'SovitsTTS': SovitsTTS,
    'CosyVoiceTTS': CosyVoiceTTS,
    'TencentTTS': TencentTTS,
    'DoubaoTTS': DoubaoTTS,
    'XTTS': XTTS
}
