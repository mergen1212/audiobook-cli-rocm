"""Командная строка ttsprodj."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from .books import SUPPORTED_SUFFIXES, load_book
from .device import gpu_report, resolve_device
from .pipeline import BuildOptions, build_audiobook, safe_filename
from .tts import DEFAULT_MODEL, DEFAULT_SPEAKER, Synthesizer

CHARS_PER_SECOND = 14

log = logging.getLogger("ttsprodj")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ttsprodj",
        description="Озвучивание книг на русском языке (Silero TTS, PyTorch ROCm).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="озвучить книгу: MP3 по главам")
    synth.add_argument("book", type=Path, help=f"файл книги ({', '.join(SUPPORTED_SUFFIXES)})")
    synth.add_argument("-o", "--output", type=Path, help="папка результата (по умолчанию audiobooks/<имя книги>)")
    synth.add_argument("-s", "--speaker", default=DEFAULT_SPEAKER, help=f"голос (по умолчанию {DEFAULT_SPEAKER})")
    synth.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"модель Silero (по умолчанию {DEFAULT_MODEL})")
    synth.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                       help="gpu — ROCm-видеокарта, auto — gpu, если доступна, иначе cpu")
    synth.add_argument("--sample-rate", type=int, choices=[24000, 48000], default=48000)
    synth.add_argument("--max-chars", type=int, default=400, help="максимум символов в одном фрагменте")
    synth.add_argument("--bitrate", default="96k", help="битрейт MP3")
    synth.add_argument("--models-dir", type=Path, default=Path("models"), help="где хранить модели Silero")
    synth.add_argument("--keep-cache", action="store_true", help="не удалять WAV-кэш фрагментов после сборки")
    synth.add_argument("--merge", action="store_true",
                       help="дополнительно склеить все главы в один MP3 с метками глав (нужен ffprobe)")

    info = sub.add_parser("info", help="показать главы книги и оценку длительности")
    info.add_argument("book", type=Path)

    speakers = sub.add_parser("speakers", help="показать голоса модели")
    speakers.add_argument("-m", "--model", default=DEFAULT_MODEL)
    speakers.add_argument("--models-dir", type=Path, default=Path("models"))

    sub.add_parser("check-gpu", help="проверить PyTorch, ROCm и видеокарту")
    return parser


def _info(args: argparse.Namespace) -> int:
    book = load_book(args.book)
    print(f"{book.title} — {book.author or 'автор не указан'}")
    total = 0
    for number, chapter in enumerate(book.chapters, start=1):
        total += chapter.char_count
        title = chapter.title or f"Глава {number}"
        print(f"{number:3d}. {title}  (абзацев: {len(chapter.paragraphs)}, символов: {chapter.char_count})")
    minutes = total / CHARS_PER_SECOND / 60
    print(f"Всего символов: {total}, примерная длительность: {minutes / 60:.1f} ч (≈{CHARS_PER_SECOND} симв./с)")
    return 0


def _synth(args: argparse.Namespace) -> int:
    if shutil.which("ffmpeg") is None:
        log.error("Не найден ffmpeg. Установите его (например, sudo pacman -S ffmpeg).")
        return 1
    if args.merge and shutil.which("ffprobe") is None:
        log.error("Не найден ffprobe (входит в пакет ffmpeg).")
        return 1
    device = resolve_device(args.device)
    synthesizer = Synthesizer(
        model_name=args.model,
        speaker=args.speaker,
        sample_rate=args.sample_rate,
        device=device,
        models_dir=args.models_dir,
    )
    output = args.output or Path("audiobooks") / safe_filename(args.book.stem)
    options = BuildOptions(
        output_dir=output,
        max_chars=args.max_chars,
        bitrate=args.bitrate,
        keep_cache=args.keep_cache,
        merge=args.merge,
    )
    files = build_audiobook(args.book, synthesizer, options)
    log.info("Готово: %d файлов в %s", len(files), output)
    return 0


def _speakers(args: argparse.Namespace) -> int:
    synthesizer = Synthesizer(
        model_name=args.model,
        speaker=None,
        sample_rate=48000,
        device=resolve_device("cpu"),
        models_dir=args.models_dir,
    )
    for speaker in synthesizer.speakers or ["(модель не сообщила список голосов)"]:
        print(speaker)
    return 0


def _check_gpu() -> int:
    lines = gpu_report()
    for line in lines:
        print(line)
    return 0 if any("GPU:" in line and "не найден" not in line for line in lines) else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = build_parser().parse_args(argv)
    handlers = {
        "synth": _synth,
        "info": _info,
        "speakers": _speakers,
        "check-gpu": lambda _: _check_gpu(),
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        log.warning("Прервано. Уже готовые фрагменты сохранены: повторный запуск продолжит работу.")
        return 130
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        log.error("Ошибка: %s", exc)
        return 1
