#!/usr/bin/env python3
"""
Step 4 variance test — compute cosine dispersion and visualize as heatmap PLY.

Run from repo root:
    python test/test_variance.py

Edit NUM_POINTS and NUM_FRAMES below. Outputs go to test/test_outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# --- config (edit these) ---
NUM_POINTS = 20_000
NUM_FRAMES = 2_000
RANDOM_SEED = 42
DEVICE = "cuda"
FORCE_RECOMPUTE = True
SHOW_VIEWER = True
# Optional: reuse precomputed artifacts (must be consistent with NUM_POINTS / frames)
POINTS_NPY: str | None = None
VISIBILITY_NPZ: str | None = None
FEATURES_DIR: str | None = None
# ---------------------------

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_DIR = TEST_DIR / "test_outputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from config import settings as cfg
from src.aggregation.variance import compute_dispersion_vectorized
from src.data.mesh import sample_mesh_points
from src.data.projection import build_visibility, load_visibility, save_visibility
from src.data.replica import load_camera_params, load_rgb, load_trajectory
from src.features.dino_backbone import extract_patch_features
from src.features.model_loaders import load_vits16
from src.features.preprocess import rgb_to_tensor
from src.utils.io import load_npy, save_npy
from src.utils.progress import tqdm
from src.viz.variance_heatmap import save_variance_histogram, save_variance_ply


def resolve_path(path_str: str | None, default: Path) -> Path:
    if path_str is None:
        return default
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def run_dir_name() -> str:
    return f"variance_{NUM_POINTS}pts_{NUM_FRAMES}frames"


def extract_features_for_frames(feat_dir: Path, n_frames: int, device: str, force: bool) -> None:
    feat_dir.mkdir(parents=True, exist_ok=True)
    device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    need_any = force or any(not (feat_dir / f"{fi:06d}.npy").exists() for fi in range(n_frames))
    if not need_any:
        print(f"Using cached ViT-S features in {feat_dir}")
        return

    print(f"Extracting ViT-S features for {n_frames} frames ({device})...")
    model = load_vits16(device)
    for fi in tqdm(range(n_frames), desc="ViT-S features"):
        out_path = feat_dir / f"{fi:06d}.npy"
        if out_path.exists() and not force:
            continue
        rgb = load_rgb(fi)
        tensor = rgb_to_tensor(rgb).to(device)
        feat = extract_patch_features(model, tensor)
        save_npy(out_path, feat)


def compute_dispersion(
    vis: dict[str, np.ndarray],
    feature_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return compute_dispersion_vectorized(
        vis,
        feature_dir=feature_dir,
        compute_pairwise=getattr(cfg, "COMPUTE_PAIRWISE", False),
    )


def show_open3d(ply_path: Path) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — skip viewer")
        return
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"  Open3D: {ply_path.name} ({len(pcd.points)} points) — close window to exit")
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Variance heatmap",
            width=1280,
            height=720,
        )
    except Exception as exc:
        print(f"  Viewer failed ({exc}). Open PLY in MeshLab: {ply_path}")


def main() -> None:
    out_dir = OUTPUT_DIR / run_dir_name()
    out_dir.mkdir(parents=True, exist_ok=True)

    points_path = resolve_path(POINTS_NPY, out_dir / f"points_{NUM_POINTS}.npy")
    vis_path = resolve_path(VISIBILITY_NPZ, out_dir / "visibility.npz")
    feat_dir = resolve_path(FEATURES_DIR, out_dir / "features_vits16")

    disp_path = out_dir / "dispersion.npy"
    pw_path = out_dir / "pairwise.npy"
    nobs_path = out_dir / "num_observations.npy"
    ply_path = out_dir / "variance_heatmap.ply"
    hist_path = out_dir / "dispersion_histogram.png"

    # --- Step 1: points ---
    if points_path.exists() and not FORCE_RECOMPUTE:
        points = load_npy(points_path)
        print(f"Loaded {len(points)} points from {points_path}")
    else:
        print(f"Sampling {NUM_POINTS} mesh vertices...")
        points, _, _ = sample_mesh_points(NUM_POINTS, RANDOM_SEED)
        save_npy(points_path, points)

    if len(points) != NUM_POINTS:
        print(f"  note: loaded {len(points)} points (NUM_POINTS={NUM_POINTS})")

    # --- Step 2: visibility ---
    if vis_path.exists() and not FORCE_RECOMPUTE:
        vis = load_visibility(vis_path)
        print(f"Loaded visibility from {vis_path}")
    else:
        cam = load_camera_params()
        poses = load_trajectory(num_frames=NUM_FRAMES)
        print(f"Building visibility: {len(points)} points x {len(poses)} frames...")
        vis = build_visibility(points, poses, cam)
        save_visibility(vis, vis_path)

    n_vis_pts = int(vis["num_points"][0])
    if n_vis_pts != len(points):
        raise ValueError(
            f"Visibility has {n_vis_pts} points but points array has {len(points)}. "
            "Use FORCE_RECOMPUTE=True or consistent artifact paths."
        )

    # --- Step 3: ViT-S features (frames 0 .. NUM_FRAMES-1) ---
    n_frames = int(vis["num_frames"][0])
    extract_features_for_frames(feat_dir, n_frames, DEVICE, force=FORCE_RECOMPUTE)

    # --- Step 4: variance ---
    if disp_path.exists() and not FORCE_RECOMPUTE:
        dispersion = load_npy(disp_path)
        pairwise = load_npy(pw_path)
        num_obs = load_npy(nobs_path)
        print(f"Loaded dispersion from {disp_path}")
    else:
        dispersion, pairwise, num_obs = compute_dispersion(vis, feat_dir)
        save_npy(disp_path, dispersion)
        save_npy(pw_path, pairwise)
        save_npy(nobs_path, num_obs)
        print(f"Dispersion: mean={dispersion.mean():.4f}, max={dispersion.max():.4f}")

    # --- Viz ---
    save_variance_ply(points, dispersion, ply_path)
    save_variance_histogram(dispersion, hist_path)

    summary = {
        "num_points": int(len(points)),
        "num_frames": n_frames,
        "num_observations_mean": float(num_obs.mean()),
        "num_observations_max": int(num_obs.max()),
        "dispersion_mean": float(dispersion.mean()),
        "dispersion_max": float(dispersion.max()),
        "dispersion_std": float(dispersion.std()),
        "legend": "hot colormap: low=dark, high=bright (more semantic flicker)",
        "outputs": {
            "heatmap_ply": str(ply_path),
            "histogram_png": str(hist_path),
            "dispersion_npy": str(disp_path),
            "visibility_npz": str(vis_path),
            "features_dir": str(feat_dir),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  {ply_path.name}       — hot colormap PLY")
    print(f"  {hist_path.name} — dispersion distribution")
    print(f"  avg observations/point: {num_obs.mean():.2f}")

    if SHOW_VIEWER:
        show_open3d(ply_path)


if __name__ == "__main__":
    main()
