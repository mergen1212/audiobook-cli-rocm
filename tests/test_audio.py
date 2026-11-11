import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ttsprodj.audio import audio_duration, merge_mp3

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="нужны ffmpeg и ffprobe",
)


def _make_mp3(path: Path, seconds: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-codec:a", "libmp3lame", "-b:a", "96k", str(path)],
        check=True,
    )


def test_merge_mp3_joins_chapters_with_markers(tmp_path: Path):
    first = tmp_path / "001.mp3"
    second = tmp_path / "002.mp3"
    _make_mp3(first, 1.0)
    _make_mp3(second, 2.0)
    target = tmp_path / "book.mp3"

    merge_mp3(
        [("Глава 1 = начало", first), ("Глава 'два'; конец", second)],
        target,
        metadata={"title": "Книга; тест", "album": "Книга; тест", "artist": "Автор"},
    )

    assert abs(audio_duration(target) - 3.0) < 0.2
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(target)],
        capture_output=True, check=True,
    )
    chapters = json.loads(output.stdout)["chapters"]
    assert [chapter["tags"]["title"] for chapter in chapters] == ["Глава 1 = начало", "Глава 'два'; конец"]
    assert not list(tmp_path.glob("*.part"))

