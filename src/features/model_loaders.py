"""Direct local checkpoint loading (avoids torch.hub cache copy)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import torch

from config import settings as cfg

_DINOV3_ROOT = Path(cfg.DINOV3_REPO).resolve()
if str(_DINOV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_DINOV3_ROOT))


def load_state_dict(weights_path: str) -> dict:
    try:
        return torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location="cpu")


def load_vits16(device: str | None = None) -> torch.nn.Module:
    from dinov3.hub.backbones import dinov3_vits16

    device = device or cfg.DEVICE
    model = dinov3_vits16(pretrained=False)
    model.load_state_dict(load_state_dict(cfg.VITS16_WEIGHTS), strict=True)
    return model.eval().to(device)


def load_vitl16(device: str | None = None) -> torch.nn.Module:
    from dinov3.hub.backbones import dinov3_vitl16

    device = device or cfg.DEVICE
    model = dinov3_vitl16(pretrained=False)
    model.load_state_dict(load_state_dict(cfg.VITL16_WEIGHTS), strict=True)
    return model.eval().to(device)


def load_dinotxt(device: str | None = None):
    from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

    device = device or cfg.DEVICE
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=False)
    model.visual_model.backbone = load_vitl16(device="cpu")
    state_dict = load_state_dict(cfg.DINOTXT_WEIGHTS)
    model.load_state_dict(state_dict, strict=False)
    return model.eval().to(device), tokenizer
