"""MIDI → OGG Opus 渲染（移植自 midi-player skill 的 synth.py）。

管线：mido 读 MIDI → 44.1kHz 纯 Python 正弦波合成（SineSynth）→
      scipy 重采样到 24kHz → soundfile 写 OGG Opus（体积约为 WAV 的 5%）。

CLI 用法（便于独立调试）:
  python midi_synth.py <in.mid> <out.ogg> [preset]
"""
import sys
from pathlib import Path

import mido
import numpy as np
from scipy import signal
import soundfile as sf

try:
    from .synth import SineSynth  # 包方式加载（插件内）
except ImportError:  # CLI 独立运行
    from synth import SineSynth  # type: ignore

PRESETS = ("piano", "guitar", "nylon", "organ", "8bit", "square",
           "strings", "bell", "flute", "default")


class MidiRenderError(Exception):
    pass


def midi_duration(mid_path) -> float:
    """MIDI 总时长（秒）。文件损坏/格式异常时抛可读 MidiRenderError。"""
    try:
        mid = mido.MidiFile(mid_path)
        return float(getattr(mid, "length", 0.0))
    except Exception as e:
        raise MidiRenderError(f"谱子文件损坏或格式异常，无法解析：{e}") from e


def repair_midi_file(src, dst) -> bool:
    """尝试修复损坏的 MIDI 文件，重写为标准文件。修复成功（可被 mido 完整解析）返回 True。

    策略（最小侵入）：
    - 能被 mido 完整解析的轨道**原样保留**（不动正常数据）；
    - 解析失败的轨道做重建：channel voice 事件的非法 data byte(>0x7F) 掩码为 0x7F，
      meta/sysex 数据原样保留（可能含 UTF-8 文本），结构彻底损坏时截断并补 End of Track。

    典型场景：个别事件的 data byte 越界（如 velocity=139），chunk 结构本身正常，
    mido 严格拒绝但内容绝大部分可用——修复后即可正常渲染。
    """
    import struct
    from io import BytesIO
    from mido.midifiles.midifiles import read_track

    try:
        data = Path(src).read_bytes()
        if data[:4] != b"MThd" or len(data) < 14:
            return False
        out = bytearray(data[:14])
        pos = 14
        fixed_any = False
        while pos + 8 <= len(data):
            tag = data[pos : pos + 4]
            if tag != b"MTrk":
                out += data[pos:]  # 非标准 chunk：原样保留到文件尾
                break
            ln = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
            full = data[pos : pos + 8 + ln]
            try:
                read_track(BytesIO(full))
                out += full  # 正常轨道原样保留
            except Exception:
                fixed_any = True
                rebuilt = _rebuild_track(data[pos + 8 : pos + 8 + ln])
                out += b"MTrk" + struct.pack(">I", len(rebuilt)) + rebuilt
            pos += 8 + ln
        if not fixed_any:
            return False  # 没有损坏轨道，无需修复
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(bytes(out))
        midi_duration(dst)  # 修复后必须能完整解析
        return True
    except Exception:
        return False


def _rebuild_track(chunk: bytes) -> bytes:
    """重建单个 MTrk 数据体：channel voice data byte 掩码，meta/sysex 原样，损坏处截断补 EOT。"""
    import struct

    new_chunk = bytearray()
    p = 0
    running = None
    try:
        while p < len(chunk):
            # delta time（VLQ，原样拷贝，其 continuation 字节本就是 >=0x80）
            dt_start = p
            while chunk[p] >= 0x80:
                p += 1
            p += 1
            new_chunk += chunk[dt_start:p]
            if p >= len(chunk):
                break
            b = chunk[p]
            if b >= 0x80:  # status byte
                status = b
                p += 1
                new_chunk.append(status)
                if status == 0xFF:  # meta：type+VLQ长度+数据 原样拷贝
                    mtype_pos = p
                    mtype = chunk[p]
                    p += 1
                    ln_start = p
                    while chunk[p] >= 0x80:
                        p += 1
                    p += 1
                    lval = _vlq_value(chunk[ln_start:p])
                    if p + lval > len(chunk):
                        break  # 溢出：截断
                    new_chunk += chunk[mtype_pos : p + lval]
                    p += lval
                    running = None
                    if mtype == 0x2F:  # End of Track
                        break
                elif status in (0xF0, 0xF7):  # sysex：长度+数据 原样
                    ln_start = p
                    while chunk[p] >= 0x80:
                        p += 1
                    p += 1
                    lval = _vlq_value(chunk[ln_start:p])
                    if p + lval > len(chunk):
                        break
                    new_chunk += chunk[ln_start : p + lval]
                    p += lval
                    running = None
                else:  # channel voice：data byte 掩码
                    running = status
                    n = 2 if (0x80 <= status <= 0xBF or status >= 0xE0) else 1
                    if p + n > len(chunk):
                        break
                    for _ in range(n):
                        db = chunk[p]
                        p += 1
                        new_chunk.append(db & 0x7F if db > 0x7F else db)
            else:  # running status
                if running is None:
                    break  # 结构损坏：截断
                n = 2 if (0x80 <= running <= 0xBF or running >= 0xE0) else 1
                new_chunk.append(b & 0x7F if b > 0x7F else b)
                p += 1
                for _ in range(n - 1):
                    if p >= len(chunk):
                        break
                    db = chunk[p]
                    p += 1
                    new_chunk.append(db & 0x7F if db > 0x7F else db)
    except IndexError:
        pass  # 截断，接受不完整轨道
    if not new_chunk.endswith(bytes([0x00, 0xFF, 0x2F, 0x00])):
        new_chunk += bytes([0x00, 0xFF, 0x2F, 0x00])  # 补 EOT
    return bytes(new_chunk)


def _vlq_value(bs: bytes) -> int:
    v = 0
    for b in bs:
        v = (v << 7) | (b & 0x7F)
    return v


def render(mid_path, out_ogg, preset: str = "piano",
           internal_sr: int = 44100, out_sr: int = 24000) -> tuple[str, float]:
    """渲染 MIDI 为 OGG Opus，返回 (输出路径, 时长秒)。CPU 密集，调用方应放线程池。"""
    if preset not in PRESETS:
        preset = "default"
    try:
        mid = mido.MidiFile(mid_path)
        duration = float(getattr(mid, "length", 0.0))
        synth = SineSynth(internal_sr, preset=preset)
        blocks = []
        for msg in mid:
            if msg.time > 0:
                n = int(msg.time * internal_sr)
                while n > 0:
                    chunk = min(n, 2048)
                    blocks.append(synth.render(chunk))
                    n -= chunk
            if msg.type == "note_on" and msg.velocity > 0:
                synth.noteon(msg.channel, msg.note, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                synth.noteoff(msg.channel, msg.note)
        blocks.append(synth.render(int(1.0 * internal_sr)))  # 1 秒尾音

        audio = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]

        if out_sr != internal_sr:
            n = int(len(audio) * out_sr / internal_sr)
            audio = signal.resample(audio, n)

        Path(out_ogg).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_ogg, audio, out_sr, format="OGG", subtype="OPUS")
        return str(out_ogg), duration
    except MidiRenderError:
        raise
    except Exception as e:
        raise MidiRenderError(f"渲染失败：{e}") from e


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: midi_synth.py <in.mid> <out.ogg> [preset]")
        sys.exit(1)
    preset = sys.argv[3] if len(sys.argv) > 3 else "piano"
    out_path, dur = render(sys.argv[1], sys.argv[2], preset)
    print(f"OK: {out_path} duration={dur:.1f}s preset={preset}")
