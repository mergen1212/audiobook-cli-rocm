"""Сборка аудиокниги: текст книги -> фрагменты -> речь -> MP3 по главам."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import encode_mp3, float_to_pcm16, merge_mp3, read_wav, silence, write_wav
from .books import Book, Chapter, load_book
from .text import Chunk, split_into_chunks
from .tts import Synthesizer

log = logging.getLogger(__name__)

PAUSE_AFTER_CHUNK = 0.15
PAUSE_AFTER_PARAGRAPH = 0.6
PAUSE_AFTER_CHAPTER = 1.0


@dataclass(frozen=True)
class BuildOptions:
    output_dir: Path
    max_chars: int = 400
    bitrate: str = "96k"
    keep_cache: bool = False
    merge: bool = False


def safe_filename(name: str, limit: int = 80) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", name)
    cleaned = " ".join(cleaned.split()).strip(" .")
    return (cleaned or "chapter")[:limit]


def build_audiobook(book_path: Path, synth: Synthesizer, options: BuildOptions) -> list[Path]:
    book = load_book(book_path)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = options.output_dir / ".cache"
    total = len(book.chapters)
    log.info('Книга «%s», автор: %s, глав: %d', book.title, book.author or "не указан", total)

    results: list[Path] = []
    for number, chapter in enumerate(book.chapters, start=1):
        title = _chapter_title(chapter, number)
        target = options.output_dir / f"{number:03d} - {safe_filename(title)}.mp3"
        if target.exists():
            log.info("[%d/%d] уже готово: %s", number, total, target.name)
            results.append(target)
            continue

        chunks = split_into_chunks([title, *chapter.paragraphs], max_chars=options.max_chars)
        log.info("[%d/%d] %s — фрагментов: %d", number, total, title, len(chunks))
        cache_dir = cache_root / f"{number:03d}"
        metadata = {
            "title": title,
            "album": book.title,
            "artist": book.author,
            "track": f"{number}/{total}",
        }
        encode_mp3(
            _chapter_audio(chunks, synth, cache_dir),
            target,
            sample_rate=synth.sample_rate,
            metadata=metadata,
            bitrate=options.bitrate,
        )
        results.append(target)
        if not options.keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)

    if options.merge:
        _merge_book(book, results, options.output_dir)

    if not options.keep_cache:
        try:
            cache_root.rmdir()
        except OSError:
            pass
    return results


def _merge_book(book: Book, files: list[Path], output_dir: Path) -> None:
    target = output_dir / f"{safe_filename(book.title)}.mp3"
    if target.exists():
        log.info("книга уже объединена: %s", target.name)
        return
    parts = [
        (_chapter_title(chapter, number), path)
        for number, (chapter, path) in enumerate(zip(book.chapters, files), start=1)
    ]
    log.info("объединяю главы в %s", target.name)
    merge_mp3(parts, target, metadata={"title": book.title, "album": book.title, "artist": book.author})


def _chapter_title(chapter: Chapter, number: int) -> str:
    return chapter.title or f"Глава {number}"


def _chapter_audio(chunks: list[Chunk], synth: Synthesizer, cache_dir: Path) -> Iterator[np.ndarray]:
    rate = synth.sample_rate
    for chunk in chunks:
        path = cache_dir / f"{_chunk_key(synth, chunk.text)}.wav"
        if path.exists():
            pcm = read_wav(path)
        else:
            pcm = float_to_pcm16(synth.synthesize(chunk.text))
            write_wav(path, pcm, rate)
        yield pcm
        pause = PAUSE_AFTER_PARAGRAPH if chunk.end_of_paragraph else PAUSE_AFTER_CHUNK
        yield silence(pause, rate)
    yield silence(PAUSE_AFTER_CHAPTER, rate)


def _chunk_key(synth: Synthesizer, text: str) -> str:
    raw = f"{synth.model_name}|{synth.speaker}|{synth.sample_rate}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
