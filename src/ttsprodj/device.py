"""Выбор устройства (GPU через ROCm или CPU) и диагностика."""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "gpu":
        raise RuntimeError("GPU недоступен: PyTorch не видит устройство ROCm. Запустите «ttsprodj check-gpu».")
    log.warning("GPU не найден, синтез пойдёт на CPU (будет медленнее)")
    return torch.device("cpu")


def gpu_report() -> list[str]:
    lines = [
        f"PyTorch: {torch.__version__}",
        f"ROCm/HIP: {torch.version.hip or 'нет (сборка PyTorch без ROCm)'}",
    ]
    if not torch.cuda.is_available():
        lines.append("GPU: не найден")
        return lines
    props = torch.cuda.get_device_properties(0)
    lines.append(f"GPU: {props.name} ({getattr(props, 'gcnArchName', 'arch неизвестна')})")
    x = torch.randn(4096, 4096, device="cuda")
    (x @ x).sum().item()
    torch.cuda.synchronize()
    lines.append("Тест матричного умножения на GPU: OK")
    return lines
