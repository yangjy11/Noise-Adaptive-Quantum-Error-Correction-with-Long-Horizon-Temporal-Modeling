"""State-dictionary checkpoints for the public learning modules."""

import json
from pathlib import Path

import torch

from .config import ModelConfig
from .model import QuantumS4Model


def save_checkpoint(path: str | Path, model: QuantumS4Model, config: ModelConfig, *, metadata: dict | None = None) -> Path:
    """Save tensors and primitive metadata, without serializing a model object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
    payload = {
        "format_version": 1,
        "model_config": config.to_dict(),
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "metadata": metadata,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[QuantumS4Model, ModelConfig, dict]:
    """Load this package's tensor checkpoint, not arbitrary Python objects."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Expected an aqec-s4-methods format_version=1 checkpoint")
    config = ModelConfig(**payload["model_config"])
    model = QuantumS4Model(**config.to_dict()).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, config, payload.get("metadata", {})
