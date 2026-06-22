"""DINOv3 ViT-S/L backbone patch feature extraction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import numpy as np
import torch
from src.utils.progress import tqdm

from config import settings as cfg
from src.data.mesh import output_dir
from src.data.replica import load_rgb, load_trajectory
from src.features.preprocess import rgb_to_tensor
from src.utils.cache import should_skip
from src.utils.io import ensure_dir, save_npy

from src.features.model_loaders import load_vits16


def features_vits16_dir() -> Path:
    return ensure_dir(output_dir() / "features_vits16")


@torch.inference_mode()
def extract_patch_features(model: torch.nn.Module, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract normalized patch tokens reshaped to (H_p, W_p, D).

    img_tensor: 1x3xHxW on same device as model
    """
    out = model.forward_features(img_tensor)
    patches = out["x_norm_patchtokens"]  # (1, N, D)
    _, n, d = patches.shape
    h_p = cfg.PATCH_GRID_H
    w_p = cfg.PATCH_GRID_W
    assert n == h_p * w_p, f"Expected {h_p * w_p} patches, got {n}"
    feat = patches.reshape(1, h_p, w_p, d).squeeze(0).cpu().numpy()
    return feat.astype(np.float32)


def extract_all_frames(force: bool = False, device: str | None = None) -> Path:
    """Step 3: extract ViT-S/16 features for all frames."""
    out_dir = features_vits16_dir()
    poses = load_trajectory()
    n_frames = len(poses)
    device = device or cfg.DEVICE
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"

    # Check if all frames cached
    if not force:
        all_exist = all(should_skip(out_dir / f"{fi:06d}.npy", False) for fi in range(n_frames))
        if all_exist:
            print(f"All {n_frames} ViT-S feature files exist, skipping.")
            return out_dir

    model = load_vits16(device)
    for fi in tqdm(range(n_frames), desc="ViT-S features"):
        out_path = out_dir / f"{fi:06d}.npy"
        if should_skip(out_path, force):
            continue
        rgb = load_rgb(fi)
        tensor = rgb_to_tensor(rgb).to(device)
        feat = extract_patch_features(model, tensor)
        save_npy(out_path, feat)

    print(f"Saved ViT-S features to {out_dir}")
    return out_dir


def load_frame_features(fi: int, feature_dir: Path | None = None) -> np.ndarray:
    feature_dir = feature_dir or features_vits16_dir()
    return np.load(feature_dir / f"{fi:06d}.npy")


def main():
    import config.settings as settings

    settings.NUM_FRAMES = 2
    extract_all_frames(force=True, device="cpu")
    feat = load_frame_features(0)
    print(f"Frame 0 features: {feat.shape}")


if __name__ == "__main__":
    main()
