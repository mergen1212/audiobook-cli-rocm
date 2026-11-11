"""Silero TTS: загрузка модели и синтез речи."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

DEFAULT_MODEL = "v5_ru"
DEFAULT_SPEAKER = "xenia"
MODEL_URL = "https://models.silero.ai/models/tts/ru/{name}.pt"


def _ensure_model(name: str, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.pt"
    if not path.exists():
        url = MODEL_URL.format(name=name)
        log.info("Скачивание модели %s", url)
        partial = path.with_name(path.name + ".part")
        torch.hub.download_url_to_file(url, str(partial), progress=True)
        partial.replace(path)
    return path


class Synthesizer:
    def __init__(
        self,
        *,
        model_name: str,
        speaker: str | None,
        sample_rate: int,
        device: torch.device,
        models_dir: Path,
    ) -> None:
        path = _ensure_model(model_name, models_dir)
        log.info("Загрузка модели %s на устройство %s", model_name, device)
        package = torch.package.PackageImporter(str(path))
        self._model = package.load_pickle("tts_models", "model")
        self._model.to(device)
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.speaker = speaker
        if speaker is not None and self.speakers and speaker not in self.speakers:
            raise ValueError(
                f"Голос «{speaker}» нет в модели {model_name}. Доступны: {', '.join(self.speakers)}"
            )

    @property
    def speakers(self) -> list[str]:
        return list(getattr(self._model, "speakers", None) or [])

    def synthesize(self, text: str) -> np.ndarray:
        if self.speaker is None:
            raise ValueError("Не выбран голос для синтеза")
        with torch.inference_mode():
            audio = self._model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=self.sample_rate,
                put_accent=True,
                put_yo=True,
            )
        return audio.float().cpu().numpy()
