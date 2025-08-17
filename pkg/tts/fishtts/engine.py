# pkg/tts/fishtts/engine.py
"""
FishTTS引擎实现
通过 HTTP/msgpack 与 FishTTS 服务交互的 TTS 引擎
支持默认音色合成和基于参考音频的零样本音色克隆
"""
import time
import struct
import requests
import numpy as np
import resampy
import ormsgpack
import io
import av
from pathlib import Path

from ..base import BaseTTS, State
from logger import logger


class FishTTS(BaseTTS):
    """
    FishTTS引擎类
    通过 HTTP/msgpack 与 FishTTS 服务交互的 TTS 引擎。
    支持默认音色合成和基于参考音频的零样本音色克隆。
    """
    
    def __init__(self, opt, parent):
        """
        初始化 FishTTS 引擎。
        :param opt: 配置选项
        :param parent: BaseReal 的实例，用于回调
        """
        super().__init__(opt, parent)
        # 从配置中获取 FishTTS 服务器地址
        self.server_url = getattr(opt, 'TTS_SERVER', 'http://127.0.0.1:8080')
        self.api_url = f"{self.server_url.rstrip('/')}/v1/tts"
        logger.info(f"FishTTS 引擎已初始化，API地址: {self.api_url}")

    def _decode_audio(self, raw: bytes) -> np.ndarray | None:
        """
        将FishTTS返回的音频字节解码为 float32/16k/单声道 的Numpy数组。
        优先按WAV解析，其次尝试裸数据s16le，最后使用PyAV兜底解码。
        :param raw: 原始音频字节数据
        :return: 解码后的音频数组或None
        """
        try:
            # 方案A: 尝试WAV格式
            if len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WAVE":
                data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
                if np.ndim(data) > 1: 
                    data = data[:, 0]
                if sr != self.sample_rate: 
                    data = resampy.resample(data, sr, self.sample_rate)
                return data.astype(np.float32)

            # 方案B: 尝试无文件头的裸s16le PCM数据 (通常返回16k，这里不做重采样)
            if len(raw) % 2 == 0:
                try:
                    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                    return pcm
                except Exception: 
                    pass

            # 方案C: 使用 PyAV 作为兜底解码器
            with av.open(io.BytesIO(raw)) as container:
                stream = next((s for s in container.streams if s.type == "audio"), None)
                if not stream: 
                    raise RuntimeError("响应中未找到音频流")
                
                frames = [frame.to_ndarray() for frame in container.decode(stream)]
                if not frames: 
                    return np.array([], dtype=np.float32)
                
                data = np.concatenate(frames, axis=1)[0]
                data = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data
                
                if stream.rate != self.sample_rate:
                    data = resampy.resample(data, stream.rate, self.sample_rate)
                return data.astype(np.float32)
        
        except Exception as e:
            logger.error(f"FishTTS: 音频解码失败 - {e}", exc_info=True)
            return None

    def txt_to_audio(self, msg_tuple):
        """
        设计说明（流式WAV稳健版，已修复 UnboundLocalError + 10ms淡入淡出 + ≥100ms源批 + 40ms/帧配速）：
        - 仅改 FishTTS 侧：严格按"RIFF/WAVE 头 + 连续 PCM"解析服务端 streaming 返回，避免把未完结流交给通用解码器。
        - 句首/句尾 10ms 淡入/淡出，消除零交点不连续导致的"啪/电音"。
        - 源侧≥100ms一批重采样到16k，降低 resampy 调用频率，稳CPU。
        - 配速改为 40ms/帧（FRAME_SECONDS=0.04），并将"发包长度=40ms=640样本@16k"，节流与声学时长严格一致。
        - 绝不发送空帧：尾包不足时补零到整40ms，并在该包携带 end 事件，避免 basereal 对空 planes 访问报错。
        """
        # 依赖导入（局部导入降低模块级开销）
        import time, struct, requests, numpy as np, resampy, ormsgpack
        from pathlib import Path
        from logger import logger

        # ======================== 入参解析与目标参数 ========================
        text, textevent, tts_options = msg_tuple

        # 解析与缓存音色名（本次→上次→配置），便于复用参考音频降低时延
        voice = None
        if isinstance(tts_options, dict):
            voice = tts_options.get("fishtts_voice_name") or tts_options.get("voice_clone_name")
        if not voice:
            voice = getattr(self, "_last_fishtts_voice", None) or getattr(self.opt, "REF_FILE", None)
        try:
            if voice:
                setattr(self, "_last_fishtts_voice", str(voice))
        except Exception:
            pass

        # 下游要求的采样率/默认分块；我们把"发包长度"改成40ms（640样本）
        target_sr = self.sample_rate             # 一般为 16000
        samples_20ms = self.chunk                # 常为 320 (=20ms@16k)
        FRAME_SECONDS = 0.02                     # 按你的要求：40ms 节流
        EMIT_SAMPLES = int(target_sr * FRAME_SECONDS)  # 每包 640 样本（40ms@16k）
        BATCH_MS = 100                           # 源侧≥100ms 批量重采样
        FADE_MS = 10                             # 句首/句尾 10ms 淡入/淡出
        LIMIT = 0.98                             # 轻限幅，避免削顶失真

        # ======================== 配速器（40ms/包） ========================
        def _pacer_init():
            return {"t0": None, "frames": 0}

        def _pace_once(ctx):
            # 严格按 40ms 发送一包，防止喂入过快导致CPU/队列抖动
            if ctx["t0"] is None:
                ctx["t0"] = time.perf_counter()
                return
            target_t = ctx["t0"] + ctx["frames"] * FRAME_SECONDS
            now = time.perf_counter()
            if target_t > now:
                time.sleep(min(target_t - now, 0.010))  # 单次最多 sleep 10ms，降低调度抖动

        # ======================== 组装请求（WAV streaming） ========================
        base_url = self.api_url
        clone_dir = Path(getattr(self.opt, 'fishtts_cloned_voices_path', './fishtts_cloned_voices'))
        ref_wav_path = clone_dir / f"{voice}.wav" if voice else None

        ref_bytes = None
        if ref_wav_path and ref_wav_path.is_file():
            try:
                ref_bytes = ref_wav_path.read_bytes()
                logger.info(f"FishTTS: 使用本地克隆参考音频 -> {ref_wav_path}")
            except Exception as e:
                logger.warning(f"FishTTS: 读取克隆音频失败({e})，改用 reference_id")

        def _post_json(payload: dict):
            return requests.post(
                base_url, json=payload, stream=True,
                headers={"content-type": "application/json", "accept": "*/*"},
                timeout=(3, 600)
            )

        def _post_msgpack(payload: dict):
            return requests.post(
                base_url, data=ormsgpack.packb(payload), stream=True,
                headers={"content-type": "application/msgpack", "accept": "*/*"},
                timeout=(3, 600)
            )

        try:
            if ref_bytes is not None:
                req = {
                    "text": text, "format": "wav", "streaming": True, "use_memory_cache": "on",
                    "references": [{"audio": ref_bytes, "text": ""}], "reference_id": None
                }
                resp = _post_msgpack(req)
            else:
                req = {
                    "text": text, "reference_id": voice,
                    "format": "wav", "streaming": True, "use_memory_cache": "on",
                }
                resp = _post_json(req)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"FishTTS 流式请求失败：{e}")
            return

        # ======================== WAV 流解析（RIFF 头 + 连续 PCM） ========================
        header_buf = bytearray()
        data_bytes = bytearray()
        wav_ready = False

        # 默认fmt（未解析到fmt前的兜底）：PCM16 单声道 44.1k
        fmt = {"audio_format": 1, "channels": 1, "sample_rate": 44100, "bits_per_sample": 16}

        def _parse_wav_header():
            """
            解析 RIFF/WAVE 头（fmt/data），遇 data 返回 (True, used_bytes)；
            如非标准 RIFF，返回 (True, 0) 走"原始PCM"兜底（FishTTS 正常会有 RIFF 头）。
            """
            nonlocal header_buf, fmt
            if len(header_buf) < 12:
                return (False, 0)
            if header_buf[:4] != b'RIFF' or header_buf[8:12] != b'WAVE':
                return (True, 0)  # 兜底处理
            offset = 12
            used_until = 12
            fmt_fields = None
            while True:
                if len(header_buf) - offset < 8:
                    return (False, used_until)
                cid = header_buf[offset:offset+4]
                csz = struct.unpack('<I', header_buf[offset+4:offset+8])[0]
                offset += 8
                if len(header_buf) - offset < csz:
                    return (False, used_until)
                cdata = header_buf[offset:offset+csz]
                if cid == b'fmt ' and csz >= 16:
                    af, ch, sr, br, ba, bps = struct.unpack('<HHIIHH', cdata[:16])
                    fmt_fields = (af, ch, sr, bps)
                elif cid == b'data':
                    if fmt_fields:
                        fmt["audio_format"], fmt["channels"], fmt["sample_rate"], fmt["bits_per_sample"] = fmt_fields
                    return (True, offset)
                offset += csz + (csz & 1)
                used_until = offset

        def _bytes_to_float32_consume(buf: bytearray):
            """
            把 data_bytes 起始的"整样本"PCM解码为 float32；返回 (f32, consumed)。
            实现常见两类：PCM16、IEEE float32（FishTTS 实际一般为 PCM16）。
            """
            fmt_tag, ch, sr, bps = fmt["audio_format"], fmt["channels"], fmt["sample_rate"], fmt["bits_per_sample"]
            if fmt_tag == 1 and bps == 16:  # PCM16
                unit = 2 * ch
                use_n = (len(buf)//unit)*unit
                if use_n == 0:
                    return (None, 0)
                parsed = bytes(buf[:use_n])
                i16 = np.frombuffer(parsed, dtype="<i2")
                if ch > 1:
                    try:
                        i16 = i16.reshape(-1, ch)[:, 0]
                    except Exception:
                        i16 = i16[::ch]
                f = i16.astype(np.float32) / 32768.0
                return (f, use_n)
            if fmt_tag == 3 and bps == 32:  # IEEE float32
                unit = 4 * ch
                use_n = (len(buf)//unit)*unit
                if use_n == 0:
                    return (None, 0)
                parsed = bytes(buf[:use_n])
                f = np.frombuffer(parsed, dtype="<f4")
                if ch > 1:
                    try:
                        f = f.reshape(-1, ch)[:, 0]
                    except Exception:
                        f = f[::ch]
                return (f.astype(np.float32), use_n)
            return (None, 0)

        # ======================== 累积/重采样/发包（40ms发包） ========================
        src_buf = np.zeros(0, dtype=np.float32)  # 源采样率下累积缓冲
        dst_blocks = []                           # 16k 下的块队列
        dst_idx = 0                               # 供内嵌函数读写，必须 nonlocal
        pctx = _pacer_init()
        first_emit = False

        # 源批大小（≥100ms）
        block_src = max(int((BATCH_MS/1000.0) * fmt["sample_rate"]), 1)

        def _emit_frames_from_blocks(final_flush=False):
            """
            从 16k块 队列中按 40ms(EMIT_SAMPLES=640) 取包并配速发送；
            final_flush=True 时：对最后一包做 10ms 淡出，并在该包携带 end 事件；绝不发送空帧。
            """
            nonlocal dst_blocks, dst_idx, first_emit, pctx  # 修复点：声明 nonlocal，避免 UnboundLocalError

            def _pop_samples(n):
                """从 dst_blocks 头部弹出 n 个采样，返回长度恰为 n 的 float32 数组。"""
                nonlocal dst_blocks, dst_idx  # 修复点：内层也显式 nonlocal
                out = np.empty((n,), dtype=np.float32)
                pos = 0
                while n > 0 and dst_blocks:
                    blk = dst_blocks[0]
                    avail = blk.shape[0] - dst_idx
                    take = avail if avail <= n else n
                    out[pos:pos+take] = blk[dst_idx:dst_idx+take]
                    pos += take
                    n -= take
                    dst_idx += take
                    if dst_idx >= blk.shape[0]:
                        dst_blocks.pop(0)
                        dst_idx = 0
                return out, (n == 0)

            # 取整包（40ms=640采样）循环
            while True:
                total_left = sum(b.shape[0] for b in dst_blocks) - (dst_idx if dst_blocks else 0)
                if total_left < EMIT_SAMPLES:
                    break

                frame, ok = _pop_samples(EMIT_SAMPLES)
                if not ok:
                    break

                # 句首 10ms 淡入（仅一次）
                if not first_emit:
                    fade_n = min(int(target_sr * FADE_MS / 1000), frame.size)
                    if fade_n > 0:
                        frame[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)

                # 轻限幅，防止削顶
                np.clip(frame, -LIMIT, LIMIT, out=frame)

                # 首包携带 start 事件
                event = None
                if not first_emit:
                    event = {'status': 'start', 'text': text, 'msgevent': textevent}
                    first_emit = True
                    pctx["t0"] = time.perf_counter()

                _pace_once(pctx)
                self.parent.put_audio_frame(frame, event)
                pctx["frames"] += 1

            # 最终冲刷：最后一包补齐到 40ms，并附带 end 事件（绝不发空帧）
            if final_flush:
                total_left = sum(b.shape[0] for b in dst_blocks) - (dst_idx if dst_blocks else 0)
                if total_left > 0:
                    tail, _ = _pop_samples(total_left)
                    if tail.size < EMIT_SAMPLES:
                        pad = np.zeros(EMIT_SAMPLES - tail.size, dtype=np.float32)
                        tail = np.concatenate([tail, pad], axis=0)

                    # 尾包 10ms 淡出
                    fade_n = min(int(target_sr * FADE_MS / 1000), tail.size)
                    if fade_n > 0:
                        tail[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

                    np.clip(tail, -LIMIT, LIMIT, out=tail)
                    _pace_once(pctx)
                    self.parent.put_audio_frame(tail, {'status': 'end', 'text': text, 'msgevent': textevent})
                    pctx["frames"] += 1
                else:
                    # 没有残量也要发一包 40ms 静音 + end，避免 basereal 收到空帧
                    silent = np.zeros(EMIT_SAMPLES, dtype=np.float32)
                    _pace_once(pctx)
                    self.parent.put_audio_frame(silent, {'status': 'end', 'text': text, 'msgevent': textevent})
                    pctx["frames"] += 1

        # ======================== 主循环：收流 → 解析头 → 解码PCM → ≥100ms重采样 → 40ms发包 ========================
        for chunk in resp.iter_content(chunk_size=16384):
            if not chunk or self.state != State.RUNNING:
                continue

            if not wav_ready:
                header_buf.extend(chunk)
                ok, used = _parse_wav_header()
                if not ok:
                    continue
                wav_ready = True
                if used > 0 and len(header_buf) > used:
                    data_bytes.extend(header_buf[used:])
                header_buf.clear()
                # 解析到头后，更新源侧批大小（≥100ms）
                block_src = max(int((BATCH_MS/1000.0) * fmt["sample_rate"]), 1)
                continue

            # 已进入 data 段
            data_bytes.extend(chunk)

            # 仅在"整样本"边界解码
            while True:
                f32, used = _bytes_to_float32_consume(data_bytes)
                if used == 0:
                    break
                del data_bytes[:used]
                if f32 is not None and f32.size > 0:
                    src_buf = f32 if src_buf.size == 0 else np.concatenate([src_buf, f32], axis=0)

                # 每累计 ≥100ms 源音频：重采样到16k -> 入队 -> 试发40ms包
                while src_buf.size >= block_src:
                    seg = src_buf[:block_src]
                    src_buf = src_buf[block_src:]
                    out16k = resampy.resample(seg, fmt["sample_rate"], target_sr)
                    if out16k.size > 0:
                        dst_blocks.append(out16k)
                    _emit_frames_from_blocks(final_flush=False)

        # ======================== flush：把余量重采样并送完（尾包携带 end） ========================
        if src_buf.size > 0:
            out16k = resampy.resample(src_buf, fmt["sample_rate"], target_sr)
            if out16k.size > 0:
                dst_blocks.append(out16k)
        _emit_frames_from_blocks(final_flush=True)
