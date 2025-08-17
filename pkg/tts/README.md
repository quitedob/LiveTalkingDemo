# TTS引擎模块

本模块包含了LiveTalking项目的所有TTS（文本转语音）引擎实现。

## 目录结构

```
pkg/tts/
├── __init__.py          # 主包入口，导出所有TTS引擎
├── requirements.txt     # 依赖包列表
├── README.md           # 本文档
├── base/               # 基础模块
│   ├── __init__.py
│   └── common.py       # BaseTTS基类和公共工具
├── edgetts/            # EdgeTTS引擎（微软Edge TTS）
│   ├── __init__.py
│   └── engine.py
├── fishtts/            # FishTTS引擎（支持音色克隆）
│   ├── __init__.py
│   └── engine.py
├── sovits/             # GPT-SoVITS引擎
│   ├── __init__.py
│   └── engine.py
├── cosyvoice/          # CosyVoice引擎
│   ├── __init__.py
│   └── engine.py
├── tencent/            # 腾讯云TTS引擎
│   ├── __init__.py
│   └── engine.py
├── doubao/             # 抖音豆包TTS引擎
│   ├── __init__.py
│   └── engine.py
└── xtts/               # XTTS引擎
    ├── __init__.py
    └── engine.py
```

## 使用方法

### 导入TTS引擎

```python
from pkg.tts import EdgeTTS, FishTTS, SovitsTTS
# 或导入所有引擎
from pkg.tts import TTS_ENGINES
```

### 基本使用

```python
# 创建TTS引擎实例
tts_engine = EdgeTTS(opt, parent)

# 发送文本进行合成
tts_engine.put_msg_txt("你好世界", eventpoint=None, **tts_options)

# 启动TTS处理线程
tts_engine.render(quit_event)
```

## 支持的TTS引擎

1. **EdgeTTS**: 基于微软Edge浏览器的TTS服务
2. **FishTTS**: 支持音色克隆的高质量TTS引擎
3. **SovitsTTS**: 基于GPT-SoVITS的语音合成引擎
4. **CosyVoiceTTS**: CosyVoice语音合成引擎
5. **TencentTTS**: 腾讯云语音合成服务
6. **DoubaoTTS**: 抖音豆包TTS服务
7. **XTTS**: XTTS语音合成引擎

## 扩展新引擎

1. 在`pkg/tts/`下创建新的引擎目录
2. 实现继承自`BaseTTS`的引擎类
3. 在引擎目录中创建`__init__.py`文件
4. 在主包的`__init__.py`中添加导入和导出

## 依赖安装

```bash
pip install -r pkg/tts/requirements.txt
```
