"""Аудио: PCM-буферы, WAV-кэш фрагментов и кодирование MP3 через ffmpeg."""

from __future__ import annotations

import re
import subprocess
import tempfile
import wave
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype="<i2")


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with wave.open(str(partial), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.astype("<i2", copy=False).tobytes())
    partial.replace(path)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def encode_mp3(
    chunks: Iterable[np.ndarray],
    target: Path,
    *,
    sample_rate: int,
    metadata: dict[str, str],
    bitrate: str = "96k",
) -> None:
    partial = target.with_name(target.name + ".part")
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-codec:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3",
    ]
    for key, value in metadata.items():
        command += ["-metadata", f"{key}={value}"]
    command.append(str(partial))

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for chunk in chunks:
            proc.stdin.write(chunk.astype("<i2", copy=False).tobytes())
    except BrokenPipeError:
        pass
    except BaseException:
        proc.kill()
        proc.wait()
        partial.unlink(missing_ok=True)
        raise
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    stderr = proc.stderr.read()
    code = proc.wait()
    if code != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg завершился с ошибкой: {stderr.decode(errors='replace').strip()}")
    partial.replace(target)


def audio_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe не смог прочитать {path.name}: {result.stderr.decode(errors='replace').strip()}")
    return float(result.stdout.decode().strip())


def merge_mp3(parts: list[tuple[str, Path]], target: Path, *, metadata: dict[str, str]) -> None:
    partial = target.with_name(target.name + ".part")
    lines = [";FFMETADATA1"] + [f"{key}={_ffmetadata_escape(value)}" for key, value in metadata.items()]
    start = 0
    for title, path in parts:
        end = start + round(audio_duration(path) * 1000)
        lines += ["", "[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}",
                  f"title={_ffmetadata_escape(title)}"]
        start = end

    with tempfile.TemporaryDirectory() as tmp:
        concat_list = Path(tmp) / "parts.txt"
        chapters = Path(tmp) / "chapters.ffmetadata"
        concat_list.write_text("".join(f"file '{_concat_escape(path)}'\n" for _, path in parts), encoding="utf-8")
        chapters.write_text("\n".join(lines) + "\n", encoding="utf-8")
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-i", str(chapters),
            "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1",
            "-codec", "copy", "-f", "mp3", str(partial),
        ]
        result = subprocess.run(command, capture_output=True)

    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg не смог объединить главы: {result.stderr.decode(errors='replace').strip()}")
    partial.replace(target)


def _ffmetadata_escape(value: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", value)


def _concat_escape(path: Path) -> str:
    return str(path.resolve()).replace("'", r"'\''")
