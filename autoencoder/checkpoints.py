"""Versioned checkpoint helpers shared by training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .models import build_model


CHECKPOINT_FORMAT_VERSION = 1


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a project checkpoint and validate its minimum schema."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {path}")

    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:  # PyTorch versions before weights_only was introduced
        checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} is not a dictionary.")
    required = {"format_version", "model_config", "model_state_dict"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(
            f"Checkpoint {path} is missing keys: {', '.join(missing)}"
        )
    if int(checkpoint["format_version"]) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {checkpoint['format_version']}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}."
        )
    return checkpoint


def load_model(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
):
    """Reconstruct a model exclusively from checkpoint metadata/state."""

    checkpoint = load_checkpoint(path, map_location=device)
    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically save one trusted project checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["format_version"] = CHECKPOINT_FORMAT_VERSION
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
    return path
