# /workspace/LiveTalking/ttsreal.py
# defina_CRT_SCURE.NO-WARNING
from __future__ import annotations
import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts
import re, io
import av, resampy
import ormsgpack, requests
from pathlib import Path
import os
import hmac
import hashlib
import base64
import json
import uuid

from typing import Iterator

import requests
import ormsgpack # 导入 ormsgpack
from pathlib import Path # 导入 Path

import queue
from queue import Queue
from io import BytesIO
import copy,websockets,gzip

from threading import Thread, Event
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from basereal import BaseReal

from logger import logger
class State(Enum):
    RUNNING=0
    PAUSE=1

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
    def __init__(self, opt, parent:BaseReal):
        self.opt=opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        # [修改] 统一队列变量名为 msg_queue，以匹配上下文
        self.msg_queue = Queue()
        self.state = State.RUNNING

    def flush_talk(self):
        # [修改] 统一队列变量名
        self.msg_queue.queue.clear()
        self.state = State.PAUSE

    # [修改] 核心修改点：为基类函数增加 **tts_options，以接收并传递额外参数
    def put_msg_txt(self, msg: str, eventpoint=None, **tts_options):
        """
        向 TTS 工作线程写入文本。
        msg          文本内容
        eventpoint   上游同步事件
        **tts_options  语速/情感/角色等可扩展参数
        """
        if len(msg) > 0:
            # [修改] 将包括 tts_options 在内的参数整体塞入队列，后续线程自行解析
            self.msg_queue.put((msg, eventpoint, tts_options))

    def render(self,quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self,quit_event):       
        while not quit_event.is_set():
            try:
                # [修改] 从队列中获取包含 tts_options 的完整元组
                msg_tuple = self.msg_queue.get(block=True, timeout=1)
                self.state=State.RUNNING
            except queue.Empty:
                continue
            # [修改] 将完整的元组传递给 txt_to_audio
            self.txt_to_audio(msg_tuple)
        logger.info('ttsreal thread stop')
    
    def txt_to_audio(self,msg):
        pass
    

###########################################################################################
class EdgeTTS(BaseTTS):
    def txt_to_audio(self, msg):
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

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        try:
            communicate = edge_tts.Communicate(text, voicename)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    self.input_stream.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logger.exception('edgetts')

###########################################################################################
# 功能：重写 FishTTS 类以支持动态音色克隆与半流式合成
class FishTTS(BaseTTS):
    """
    通过 HTTP/msgpack 与 FishTTS 服务交互的 TTS 引擎。
    支持默认音色合成和基于参考音频的零样本音色克隆。
    """
    def __init__(self, opt, parent: BaseReal):
        """
        初始化 FishTTS 引擎。
        - opt: 配置选项
        - parent: BaseReal 的实例，用于回调
        """
        super().__init__(opt, parent)
        # 从配置中获取 FishTTS 服务器地址
        self.server_url = getattr(opt, 'TTS_SERVER', 'http://127.0.0.1:8080')
        self.api_url = f"{self.server_url.rstrip('/')}/v1/tts"
        logger.info(f"FishTTS 引擎已初始化，API地址: {self.api_url}")

    def _decode_audio(self, raw: bytes) -> np.ndarray | None:
        """
        将FishTTS返回的音频字节解码为 float32/16k/单声道 的Numpy数组。
        - 优先按WAV解析，其次尝试裸数据s16le，最后使用PyAV兜底解码。
        """
        try:
            # 方案A: 尝试WAV格式
            if len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WAVE":
                data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
                if np.ndim(data) > 1: data = data[:, 0]
                if sr != self.sample_rate: data = resampy.resample(data, sr, self.sample_rate)
                return data.astype(np.float32)

            # 方案B: 尝试无文件头的裸s16le PCM数据 (通常返回16k，这里不做重采样)
            if len(raw) % 2 == 0:
                try:
                    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                    return pcm
                except Exception: pass

            # 方案C: 使用 PyAV 作为兜底解码器
            with av.open(io.BytesIO(raw)) as container:
                stream = next((s for s in container.streams if s.type == "audio"), None)
                if not stream: raise RuntimeError("响应中未找到音频流")
                
                frames = [frame.to_ndarray() for frame in container.decode(stream)]
                if not frames: return np.array([], dtype=np.float32)
                
                data = np.concatenate(frames, axis=1)[0]
                data = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data
                
                if stream.rate != self.sample_rate:
                    data = resampy.resample(data, stream.rate, self.sample_rate)
                return data.astype(np.float32)
        
        except Exception as e:
            logger.error(f"FishTTS: 音频解码失败 - {e}", exc_info=True)
            return None
# 文件路径：/workspace/LiveTalking/ttsreal.py
    def txt_to_audio(self, msg_tuple):
        """
        中文说明（WAV优先版/稳态+低CPU）：
        - 始终以 format='wav', streaming=True 请求服务端（不再优先 s16le@16k）。
        - 流式解析 RIFF/WAVE 头（fmt/data），自动识别 PCM/IEEE float/A-law/μ-law/8/16/24/32/64 位，正确解码为 float32 单声道。
        - 在线重采样按“源采样率下每 80ms 一块”批处理 → 16k，降低 resampy 调用频率，稳定 CPU。
        - 帧输出严格 20ms/帧配速（≈50fps），避免“推得太快挤爆下游队列”。
        - 若存在本地克隆音色 WAV（fishtts_cloned_voices/<voice>.wav），优先走 references（msgpack）；否则用 reference_id（json）。
        """
        # —— 导入依赖（函数内导入，避免模块级开销） ——
        import time, struct, requests, numpy as np, resampy, ormsgpack
        from pathlib import Path
        from logger import logger

        # ========= 基本参数 =========
        text, textevent, tts_options = msg_tuple
        # 解析音色名（本次传入优先；否则沿用上次成功；再否则用配置）
        voice = None
        if isinstance(tts_options, dict):
            voice = tts_options.get("fishtts_voice_name") or tts_options.get("voice_clone_name")
        if not voice:
            voice = getattr(self, "_last_fishtts_voice", None) or getattr(self.opt, "REF_FILE", None)

        target_sr = self.sample_rate           # 16_000
        samples_per_frame = self.chunk         # 320 (= 20ms@16k)
        FRAME_SECONDS = 0.02                   # 20ms/帧

        # —— 配速器（防止推帧过快挤爆CPU/下游） ——
        ENABLE_PACING = True
        PACE_SLEEP_CAP = 0.010                 # 单次最多 sleep 10ms

        def _pacer_init():
            return {"t0": None, "frames": 0}

        def _pace_once(pctx):
            if not ENABLE_PACING:
                return
            if pctx["t0"] is None:
                pctx["t0"] = time.perf_counter()
                return
            target_t = pctx["t0"] + pctx["frames"] * FRAME_SECONDS
            now = time.perf_counter()
            if target_t > now:
                time.sleep(min(target_t - now, PACE_SLEEP_CAP))

        # ========= 选择 references 或 reference_id =========
        # 说明：有本地克隆音色 wav 则走 references（msgpack），否则 ref_id（json）
        clone_dir = Path(getattr(self.opt, 'fishtts_cloned_voices_path', './fishtts_cloned_voices'))
        ref_wav_path = clone_dir / f"{voice}.wav" if voice else None
        ref_audio_bytes = None
        if ref_wav_path and ref_wav_path.is_file():
            try:
                ref_audio_bytes = ref_wav_path.read_bytes()
                logger.info(f"FishTTS: 使用本地克隆参考音频 -> {ref_wav_path}")
            except Exception as e:
                logger.warning(f"FishTTS: 读取克隆音频失败({e})，改用 reference_id")

        # ========= 发起“WAV流式”请求（不再尝试 s16le） =========
        base_url = self.api_url  # e.g. http://127.0.0.1:8080/v1/tts

        def _post_json(payload):
            # 使用 JSON 载荷（reference_id）
            return requests.post(base_url, json=payload, stream=True,
                                headers={"content-type": "application/json", "accept": "*/*"},
                                timeout=(3, 600))

        def _post_msgpack(payload):
            # 使用 msgpack 载荷（references: 直接带参考音频字节）
            return requests.post(base_url, data=ormsgpack.packb(payload), stream=True,
                                headers={"content-type": "application/msgpack", "accept": "*/*"},
                                timeout=(3, 600))

        try:
            if ref_audio_bytes is not None:
                # —— references（msgpack）路径：format=wav, streaming=true ——
                req = {
                    "text": text,
                    "format": "wav",
                    "streaming": True,
                    "use_memory_cache": "on",
                    "references": [{"audio": ref_audio_bytes, "text": ""}],
                    "reference_id": None
                }
                resp = _post_msgpack(req)
            else:
                # —— reference_id（json）路径：format=wav, streaming=true ——
                req = {
                    "text": text,
                    "reference_id": voice,
                    "format": "wav",
                    "streaming": True,
                    "use_memory_cache": "on",
                }
                resp = _post_json(req)

            resp.raise_for_status()
        except Exception as e:
            logger.error(f"FishTTS WAV 流式请求失败：{e}")
            return

        # 粘性音色：成功请求后记住本次 voice
        try:
            if voice:
                setattr(self, "_last_fishtts_voice", str(voice))
        except Exception:
            pass

        # ========= WAV 流稳健解析（fmt/data） =========
        header_buf = bytearray()   # 累积 RIFF/WAVE 头与 chunk 头
        wav_ready = False          # 是否已遇到 data 段
        fmt = {
            "audio_format": 1,     # 1=PCM, 3=IEEE_FLOAT, 6=A-law, 7=μ-law
            "channels": 1,
            "sample_rate": 44100,
            "bits_per_sample": 16
        }
        data_bytes = bytearray()   # data 段原始字节缓存

        # —— G.711 μ/A-law 查表（8bit -> int16） ——
        _ulaw_lut, _alaw_lut = None, None
        def _g711_tables():
            nonlocal _ulaw_lut, _alaw_lut
            if _ulaw_lut is not None:
                return _ulaw_lut, _alaw_lut
            # μ-law
            ulaw = np.zeros(256, dtype=np.int16)
            for i in range(256):
                u = ~i & 0xFF
                sign = (u & 0x80)
                exponent = (u >> 4) & 0x07
                mantissa = u & 0x0F
                sample = ((mantissa << 4) + 0x08) << (exponent + 2)
                sample -= 0x84
                ulaw[i] = -sample if sign else sample
            _ulaw_lut = ulaw
            # A-law
            alaw = np.zeros(256, dtype=np.int16)
            for i in range(256):
                a = i ^ 0x55
                sign = a & 0x80
                exponent = (a >> 4) & 0x07
                mantissa = a & 0x0F
                if exponent > 0:
                    sample = ((mantissa << 4) + 0x08 + 0x100) << (exponent - 1)
                else:
                    sample = (mantissa << 4) + 0x08
                alaw[i] = -sample if sign else sample
            _alaw_lut = alaw
            return _ulaw_lut, _alaw_lut

        def _parse_wav_header():
            """
            解析 RIFF/WAVE 头；当解析到 data 段开头时返回 (True, used_bytes)。
            - used_bytes：头部消耗的总字节数（含 data 头），剩余即为 data 起始数据。
            """
            nonlocal header_buf, fmt
            if len(header_buf) < 12:
                return (False, 0)
            if header_buf[:4] != b'RIFF' or header_buf[8:12] != b'WAVE':
                # 非标准 WAV：视为“无头”原始流，交给后续原始PCM处理
                return (True, 0)

            offset = 12
            used_until = 12
            fmt_fields = None

            while True:
                if len(header_buf) - offset < 8:
                    return (False, used_until)  # 头不足
                cid = header_buf[offset:offset+4]
                csz = struct.unpack('<I', header_buf[offset+4:offset+8])[0]
                offset += 8
                if len(header_buf) - offset < csz:
                    return (False, used_until)  # chunk 内容不足
                cdata = header_buf[offset:offset+csz]
                if cid == b'fmt ' and csz >= 16:
                    af, ch, sr, br, ba, bps = struct.unpack('<HHIIHH', cdata[:16])
                    fmt_fields = (af, ch, sr, bps)
                elif cid == b'data':
                    # 更新 fmt（若已获得）
                    if fmt_fields:
                        fmt["audio_format"], fmt["channels"], fmt["sample_rate"], fmt["bits_per_sample"] = fmt_fields
                    return (True, offset)  # data 段开始位置
                # 对齐偶数字节
                offset += csz + (csz & 1)
                used_until = offset

        # ========= 在线解码 + 80ms批量重采样 + 20ms配速输出 =========
        first_sent = False
        pace = _pacer_init()

        # 源缓冲（源采样率下的 float32）、目标块队列（16k下的块，按20ms取帧）
        src_buf = np.zeros(0, dtype=np.float32)
        dst_blocks = []   # list[np.ndarray] @16k
        dst_idx = 0

        def _emit_frames_from_blocks():
            """从 16k 缓冲块按 20ms/帧取出并配速发送（避免大规模拼接）。"""
            nonlocal first_sent, pace, dst_blocks, dst_idx
            needed = samples_per_frame
            while True:
                total = sum(b.shape[0] for b in dst_blocks) - (dst_idx if dst_blocks else 0)
                if total < needed:
                    break
                out = np.empty((needed,), dtype=np.float32)
                pos = 0
                while needed > 0:
                    blk = dst_blocks[0]
                    avail = blk.shape[0] - dst_idx
                    take = avail if avail <= needed else needed
                    out[pos:pos+take] = blk[dst_idx:dst_idx+take]
                    pos += take; needed -= take; dst_idx += take
                    if dst_idx >= blk.shape[0]:
                        dst_blocks.pop(0); dst_idx = 0
                event = None
                if not first_sent:
                    event = {'status': 'start', 'text': text, 'msgevent': textevent}
                    first_sent = True
                    pace["t0"] = time.perf_counter()
                _pace_once(pace)
                self.parent.put_audio_frame(out, event)
                pace["frames"] += 1
                needed = samples_per_frame  # 继续尝试下一帧

        # 源重采样块大小：按 80ms@源采样率
        block_src = max(int((80/1000.0) * fmt["sample_rate"]), 1)

        # 将 data_bytes 的起始若干完整样本解码为 float32，并返回(浮点数组, 消费字节数)
        def _bytes_to_float32_consume(buf: bytearray):
            fmt_tag, ch, sr, bps = fmt["audio_format"], fmt["channels"], fmt["sample_rate"], fmt["bits_per_sample"]

            if fmt_tag in (1, 3):  # PCM / IEEE float
                if bps == 8:
                    unit = 1 * ch
                    use_n = (len(buf)//unit)*unit
                    if use_n == 0: return (None, 0)
                    parsed = bytes(buf[:use_n])
                    u8 = np.frombuffer(parsed, dtype=np.uint8)
                    if ch > 1:
                        try: u8 = u8.reshape(-1, ch)[:,0]
                        except Exception: u8 = u8[::ch]
                    f = (u8.astype(np.float32) - 128.0) / 128.0
                    return (f, use_n)
                elif bps == 16:
                    unit = 2 * ch
                    use_n = (len(buf)//unit)*unit
                    if use_n == 0: return (None, 0)
                    parsed = bytes(buf[:use_n])
                    i16 = np.frombuffer(parsed, dtype="<i2")
                    if ch > 1:
                        try: i16 = i16.reshape(-1, ch)[:,0]
                        except Exception: i16 = i16[::ch]
                    f = i16.astype(np.float32) / 32768.0
                    return (f, use_n)
                elif bps == 24:
                    unit = 3 * ch
                    use_n = (len(buf)//unit)*unit
                    if use_n == 0: return (None, 0)
                    parsed = memoryview(bytes(buf[:use_n]))
                    b = np.frombuffer(parsed, dtype=np.uint8).reshape(-1, 3*ch)
                    # 仅取第1声道的三字节
                    b0 = b[:, :3]
                    i32 = (b0[:,0].astype(np.int32) | (b0[:,1].astype(np.int32)<<8) | (b0[:,2].astype(np.int32)<<16))
                    sign = (i32 & 0x800000) != 0
                    i32[sign] |= ~0xffffff  # 符号扩展
                    f = i32.astype(np.float32) / 8388608.0
                    return (f, use_n)
                elif bps == 32 and fmt_tag == 1:
                    unit = 4 * ch
                    use_n = (len(buf)//unit)*unit
                    if use_n == 0: return (None, 0)
                    parsed = bytes(buf[:use_n])
                    i32 = np.frombuffer(parsed, dtype="<i4")
                    if ch > 1:
                        try: i32 = i32.reshape(-1, ch)[:,0]
                        except Exception: i32 = i32[::ch]
                    f = i32.astype(np.float32) / 2147483648.0
                    return (f, use_n)
                elif fmt_tag == 3 and bps in (32, 64):  # IEEE float
                    unit = (4 if bps==32 else 8) * ch
                    use_n = (len(buf)//unit)*unit
                    if use_n == 0: return (None, 0)
                    parsed = bytes(buf[:use_n])
                    dt = np.float32 if bps==32 else np.float64
                    f = np.frombuffer(parsed, dtype=dt)
                    if ch > 1:
                        try: f = f.reshape(-1, ch)[:,0]
                        except Exception: f = f[::ch]
                    f = f.astype(np.float32)
                    return (f, use_n)
                else:
                    return (None, 0)

            elif fmt_tag in (6, 7):  # A-law / μ-law
                unit = 1 * ch
                use_n = (len(buf)//unit)*unit
                if use_n == 0: return (None, 0)
                parsed = bytes(buf[:use_n])
                lut_u, lut_a = _g711_tables()
                u8 = np.frombuffer(parsed, dtype=np.uint8)
                if ch > 1:
                    try: u8 = u8.reshape(-1, ch)[:,0]
                    except Exception: u8 = u8[::ch]
                i16 = (lut_u[u8] if fmt_tag == 7 else lut_a[u8])
                f = i16.astype(np.float32) / 32768.0
                return (f, use_n)

            # 其它非常见编码（如 ADPCM）——不在此解码，避免误判电音
            return (None, 0)

        # ========= 主循环：收流 → 解析头 → 解码数据 → 重采样 → 20ms配速输出 =========
        for chunk in resp.iter_content(chunk_size=16384):
            if not chunk or self.state != State.RUNNING:
                continue

            if not wav_ready:
                # 头部尚未就绪：累积并尝试解析
                header_buf.extend(chunk)
                ok, used = _parse_wav_header()
                if not ok:
                    continue  # 头不完整，继续收
                wav_ready = True
                # used 前是头；若头后面带了部分 data，则移入 data_bytes
                if used > 0 and len(header_buf) > used:
                    data_bytes.extend(header_buf[used:])
                header_buf.clear()
                # 更新“重采样块大小”为 80ms@源采样率
                block_src = max(int((80/1000.0) * fmt["sample_rate"]), 1)
                continue

            # 走到这里：已经进入 data 段
            data_bytes.extend(chunk)

            # 尝试把 data_bytes 转为 float32 源数据；只在“整样本宽度”边界上消费
            while True:
                f32, used = _bytes_to_float32_consume(data_bytes)
                if used == 0:
                    break
                # 重要：先复制 bytes 再删除原片段（上面已复制），避免 BufferError
                del data_bytes[:used]
                if f32 is not None and f32.size > 0:
                    src_buf = f32 if src_buf.size == 0 else np.concatenate([src_buf, f32])

                # 每累计 ≥80ms 源音频：重采样一次 → 16k，并尝试输出 20ms 帧
                while src_buf.size >= block_src:
                    block = src_buf[:block_src]
                    src_buf = src_buf[block_src:]
                    out16k = resampy.resample(block, fmt["sample_rate"], target_sr)
                    if out16k.size > 0:
                        dst_blocks.append(out16k)
                    _emit_frames_from_blocks()

        # ========= flush：把余量重采样并送完 =========
        if 'fmt' in locals() and src_buf.size > 0:
            out16k = resampy.resample(src_buf, fmt["sample_rate"], target_sr)
            if out16k.size > 0:
                dst_blocks.append(out16k)
        _emit_frames_from_blocks()

        # 结束事件
        if self.state == State.RUNNING:
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32),
                                        {'status': 'end', 'text': text, 'msgevent': textevent})



###########################################################################################
class SovitsTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        # [修改] 解包三元组
        text, textevent, tts_options = msg
        self.stream_tts(
            self.gpt_sovits(
                text=text,
                reffile=self.opt.REF_FILE,
                reftext=self.opt.REF_TEXT,
                language="zh", #en args.language,
                server_url=self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg # 传递完整元组
        )

    def gpt_sovits(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':language,
            'media_type':'ogg',
            'streaming_mode':True
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
        
            for chunk in res.iter_content(chunk_size=None): #12800 1280 32K*20ms*2
                logger.info('chunk len:%d',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('sovits')

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def stream_tts(self,audio_stream,msg):
        # [修改] 解包三元组
        text, textevent, tts_options = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:     
                byte_stream=BytesIO(chunk)
                stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)

###########################################################################################
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self,msg):
        # [修改] 解包三元组
        text, textevent, tts_options = msg
        self.stream_tts(
            self.cosy_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg # 传递完整元组
        )

    def cosy_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        try:
            files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]
            res = requests.request("GET", f"{server_url}/inference_zero_shot", data=payload, files=files, stream=True)
            
            end = time.perf_counter()
            logger.info(f"cosy_voice Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=9600): # 960 24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"cosy_voice Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('cosyvoice')

    def stream_tts(self,audio_stream,msg):
        # [修改] 解包三元组
        text, textevent, tts_options = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:     
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################
_PROTOCOL = "https://"
_HOST = "tts.cloud.tencent.com"
_PATH = "/stream"
_ACTION = "TextToStreamAudio"

class TencentTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.appid = os.getenv("TENCENT_APPID")
        self.secret_key = os.getenv("TENCENT_SECRET_KEY")
        self.secret_id = os.getenv("TENCENT_SECRET_ID")
        self.voice_type = int(opt.REF_FILE)
        self.codec = "pcm"
        self.sample_rate = 16000
        self.volume = 0
        self.speed = 0
    
    def __gen_signature(self, params):
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

    def txt_to_audio(self,msg):
        # [修改] 解包三元组
        text, textevent, tts_options = msg
        self.stream_tts(
            self.tencent_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg # 传递完整元组
        )

    def tencent_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
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
        
            for chunk in res.iter_content(chunk_size=6400): # 640 16K*20ms*2
                #logger.info('chunk len:%d',len(chunk))
                if first:
                    try:
                        rsp = json.loads(chunk)
                        #response["Code"] = rsp["Response"]["Error"]["Code"]
                        #response["Message"] = rsp["Response"]["Error"]["Message"]
                        logger.error("tencent tts:%s",rsp["Response"]["Error"]["Message"])
                        return
                    except:
                        end = time.perf_counter()
                        logger.info(f"tencent Time to first chunk: {end-start}s")
                        first = False                    
                if chunk and self.state==State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('tencent')

    def stream_tts(self,audio_stream,msg):
        # [修改] 解包三元组
        text, textevent, tts_options = msg
        first = True
        last_stream = np.array([],dtype=np.float32)
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:     
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = np.concatenate((last_stream,stream))
                #stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:] #get the remain stream
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################


class DoubaoTTS(BaseTTS):
    def __init__(self, opt, parent):
        # 【简介】初始化抖音豆包TTS客户端；自动根据 REF_FILE 决定集群（S_ 前缀→volcano_icl）
        super().__init__(opt, parent)
        # 读取凭证（需提前 export DOUBAO_APPID / DOUBAO_TOKEN）
        self.appid = os.getenv("DOUBAO_APPID")
        self.token = os.getenv("DOUBAO_TOKEN")

        # 【关键修复】允许通过环境变量覆盖集群；若 REF_FILE 以 "S_" 开头（克隆声线），默认用 volcano_icl
        ref = (opt.REF_FILE or "").strip()
        env_cluster = os.getenv("DOUBAO_CLUSTER", "").strip()
        if env_cluster:
            _cluster = env_cluster
        else:
            _cluster = "volcano_icl" if ref.startswith("S_") else "volcano_tts"

        # WebSocket 地址保持不变
        _host = "openspeech.bytedance.com"
        self.api_url = f"wss://{_host}/api/v1/tts/ws_binary"

        # 组装默认请求体（后面会在 doubao_voice 里覆写 voice_type / text / reqid）
        self.request_json = {
            "app": {
                "appid": self.appid,
                # 注意：这里的 token 字段不用填 access_token；真正的鉴权走 WS 头部 Authorization
                "token": "access_token",
                "cluster": _cluster,  # 这里使用上面推断/外部传入的集群
            },
            "user": { "uid": "xxx" },
            "audio": {
                "voice_type": "xxx",
                # 【说明】项目里按 PCM int16 解码流，所以保持 pcm 编码
                "encoding": "pcm",
                "rate": 16000,
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": "xxx",
                "text": "字节跳动语音合成。",
                "text_type": "plain",
                "operation": "xxx",
            },
        }

        # 【可选健壮性日志】帮助定位未设置 token 的情况（避免出现 'Bearer;'）
        if not self.token:
            logger.warning("DoubaoTTS: DOUBAO_TOKEN 未设置，WS 将以 'Bearer;' 发送，可能导致无音频返回。")
            
    # [新增] 专门为 DoubaoTTS 新增的 put_msg_txt 方法，用于解析 tts_options
    def put_msg_txt(self, msg, eventpoint=None, **tts_options):
        """覆盖父类方法，先把可映射的参数写入 self.request_json，再走父类逻辑"""
        # ---------- 把 tts_options 映射进 request_json ----------
        audio_cfg = self.request_json["audio"]
        if "speed" in tts_options:
            audio_cfg["speed_ratio"] = float(tts_options["speed"])
        if "volume" in tts_options:
            audio_cfg["volume_ratio"] = float(tts_options["volume"])
        if "pitch" in tts_options:
            audio_cfg["pitch_ratio"] = float(tts_options["pitch"])
        if "emotion" in tts_options:
            audio_cfg["enable_emotion"] = True
            audio_cfg["emotion"] = tts_options["emotion"]
        # -------------------------------------------------------
        # 调用父类方法，将消息放入队列，保持工作流一致
        super().put_msg_txt(msg, eventpoint, **tts_options)

    async def doubao_voice(self, text): # -> Iterator[bytes]:
        start = time.perf_counter()
        voice_type = self.opt.REF_FILE

        try:
            # 创建请求对象
            default_header = bytearray(b'\x11\x10\x11\x00')
            # 使用已经根据 tts_options 修改过的 request_json
            submit_request_json = copy.deepcopy(self.request_json)
            submit_request_json["user"]["uid"] = self.parent.sessionid
            submit_request_json["audio"]["voice_type"] = voice_type
            submit_request_json["request"]["text"] = text
            submit_request_json["request"]["reqid"] = str(uuid.uuid4())
            submit_request_json["request"]["operation"] = "submit"
            payload_bytes = str.encode(json.dumps(submit_request_json))
            payload_bytes = gzip.compress(payload_bytes)  # if no compression, comment this line
            full_client_request = bytearray(default_header)
            full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
            full_client_request.extend(payload_bytes)  # payload

            header = {"Authorization": f"Bearer; {self.token}"}
            first = True
            async with websockets.connect(self.api_url, extra_headers=header, ping_interval=None) as ws:
                await ws.send(full_client_request)
                while True:
                    res = await ws.recv()
                    header_size = res[0] & 0x0f
                    message_type = res[1] >> 4
                    message_type_specific_flags = res[1] & 0x0f
                    payload = res[header_size*4:]

                    if message_type == 0xb:  # audio-only server response
                        if message_type_specific_flags == 0:  # no sequence number as ACK
                            #print("                                 Payload size: 0")
                            continue
                        else:
                            if first:
                                end = time.perf_counter()
                                logger.info(f"doubao tts Time to first chunk: {end-start}s")
                                first = False
                            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                            payload = payload[8:]
                            yield payload
                        if sequence_number < 0:
                            break
                    else:
                        break
        except Exception as e:
            logger.exception('doubao')

    def txt_to_audio(self, msg):
        # [修改] 解包三元组
        text, textevent, tts_options = msg
        asyncio.new_event_loop().run_until_complete(
            self.stream_tts(
                self.doubao_voice(text),
                msg # 传递完整元组
            )
        )

    async def stream_tts(self, audio_stream, msg):
        # [修改] 解包三元组
        text, textevent, tts_options = msg
        first = True
        last_stream = np.array([],dtype=np.float32)
        async for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = np.concatenate((last_stream,stream))
                #stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                # byte_stream=BytesIO(buffer)
                # stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = None
                    if first:
                        # [修正] 修复一个可能的拼写错误 msgenvent -> msgevent
                        eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:] #get the remain stream
        # [修正] 修复一个可能的拼写错误 msgenvent -> msgevent
        eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

###########################################################################################
class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self,msg):
        # [修改] 解包三元组
        text, textevent, tts_options = msg
        self.stream_tts(
            self.xtts(
                text,
                self.speaker,
                "zh-cn", #en args.language,
                self.opt.TTS_SERVER, #"http://localhost:9000", #args.server_url,
                "20" #args.stream_chunk_size
            ),
            msg # 传递完整元组
        )

    def get_speaker(self,ref_audio,server_url):
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()

    def xtts(self,text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        try:
            res = requests.post(
                f"{server_url}/tts_stream",
                json=speaker,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"xtts Time to make POST: {end-start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True
        
            for chunk in res.iter_content(chunk_size=9600): #24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"xtts Time to first chunk: {end-start}s")
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            print(e)
    
    def stream_tts(self,audio_stream,msg):
        # [修改] 解包三元组
        text, textevent, tts_options = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:     
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)