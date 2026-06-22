#!/usr/bin/env python3
"""
DINOtxt patch feature test — PCA visualization for one frame.

Extracts DINOtxt patch tokens, assigns them to all visible mesh vertices,
PCA-colors features, and visualizes the full mesh in world 3D (gray = not
visible in this frame). Also saves a 2D upsampled PCA image of the patch grid.

Run from repo root:
    python test/test_dinotxt_pca.py

Edit FRAME_IDX below. Outputs go to test/test_outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- config (edit these) ---
FRAME_IDX = 1
EXTRACT_IF_MISSING = True
DEVICE = "cuda"
SHOW_VIEWER = True  # Open3D window for full-mesh PCA cloud
FEATURES_DIR: str | None = None  # None = test_outputs cache for this run
COLOR_NOT_VISIBLE = (80, 80, 80)
# ---------------------------

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_DIR = TEST_DIR / "test_outputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from config import settings as cfg
from src.data.mesh import load_semantic_mesh
from src.data.projection import project_frame_visibility
from src.data.replica import frame_path, load_camera_params, load_rgb, load_trajectory
from src.features.dinotxt_features import extract_aligned_patches
from src.features.model_loaders import load_dinotxt
from src.features.preprocess import pad_image_rgb, rgb_to_tensor
from src.utils.io import save_npy
from src.viz.gt_labels import save_colored_ply


def run_dir_name() -> str:
    return f"dinotxt_pca_frame_{FRAME_IDX:06d}"


def resolve_features_dir(out_dir: Path) -> Path:
    if FEATURES_DIR:
        path = Path(FEATURES_DIR)
        return path if path.is_absolute() else REPO_ROOT / path
    return out_dir


def load_or_extract_frame_features(out_dir: Path) -> np.ndarray:
    feat_dir = resolve_features_dir(out_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / f"features_dinotxt_{FRAME_IDX:06d}.npy"

    if feat_path.exists():
        print(f"Loading cached features: {feat_path}")
        return np.load(feat_path)

    if not EXTRACT_IF_MISSING:
        raise FileNotFoundError(
            f"No features at {feat_path}. Set EXTRACT_IF_MISSING=True."
        )

    device = DEVICE if DEVICE != "cuda" or torch.cuda.is_available() else "cpu"
    print(f"Extracting DINOtxt features for frame {FRAME_IDX} ({device})...")
    model, _ = load_dinotxt(device)
    rgb = load_rgb(FRAME_IDX)
    print(f"  source: {frame_path(FRAME_IDX)}")
    tensor = rgb_to_tensor(rgb).to(device)
    feat = extract_aligned_patches(model, tensor)
    save_npy(feat_path, feat)
    print(f"Saved features to {feat_path}")
    return feat


def pca_features_to_rgb(feats: np.ndarray) -> np.ndarray:
    """PCA (N, D) feature vectors to uint8 RGB (N, 3)."""
    x = feats.astype(np.float64)
    x = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pcs = x @ vt[:3].T

    rgb = np.zeros((len(feats), 3), dtype=np.float64)
    for c in range(3):
        ch = pcs[:, c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        rgb[:, c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def pca_patch_grid_to_rgb(feat: np.ndarray) -> np.ndarray:
    """PCA-color full patch grid (H, W, D) for 2D visualization."""
    h, w, d = feat.shape
    colors = pca_features_to_rgb(feat.reshape(-1, d))
    return colors.reshape(h, w, 3)


def upsample_patch_image(patch_img: np.ndarray, patch_size: int = cfg.PATCH_SIZE) -> np.ndarray:
    return np.repeat(np.repeat(patch_img, patch_size, axis=0), patch_size, axis=1)


def build_mesh_colors(n_points: int, vis_idx: np.ndarray, vis_colors: np.ndarray) -> np.ndarray:
    colors = np.tile(np.array(COLOR_NOT_VISIBLE, dtype=np.uint8), (n_points, 1))
    colors[vis_idx] = vis_colors
    return colors


def save_scatter_png(
    points: np.ndarray,
    colors: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    rgb = colors.astype(np.float32) / 255.0
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=rgb,
        s=0.15,
        alpha=0.85,
        marker=".",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_comparison_figure(
    rgb: np.ndarray,
    pca_full: np.ndarray,
    out_path: Path,
    frame_idx: int,
) -> None:
    rgb_pad = pad_image_rgb(rgb)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(rgb_pad)
    axes[0].set_title(f"Frame {frame_idx:06d} — RGB")
    axes[0].axis("off")
    axes[1].imshow(pca_full)
    axes[1].set_title(f"Frame {frame_idx:06d} — DINOtxt PCA (patch grid)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def show_open3d(ply_path: Path, title: str) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — skip viewer")
        return
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"  Open3D: {title} ({len(pcd.points):,} points) — close window to exit")
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=title,
            width=1280,
            height=720,
        )
    except Exception as exc:
        print(f"  Viewer failed ({exc}). Open PLY manually: {ply_path}")


def main() -> None:
    out_dir = OUTPUT_DIR / run_dir_name()
    out_dir.mkdir(parents=True, exist_ok=True)

    poses = load_trajectory()
    if FRAME_IDX < 0 or FRAME_IDX >= len(poses):
        raise ValueError(f"FRAME_IDX={FRAME_IDX} out of range [0, {len(poses) - 1}]")

    print(f"Loading all vertices from {cfg.MESH_PATH}...")
    vertices, _, _ = load_semantic_mesh(cfg.MESH_PATH)
    points = vertices.astype(np.float32)
    print(f"  {len(points):,} mesh vertices")

    cam = load_camera_params()
    print(f"Projecting into frame {FRAME_IDX}...")
    frame_vis = project_frame_visibility(points, FRAME_IDX, poses, cam)
    vis_idx = frame_vis["visible_indices"]
    n_visible = len(vis_idx)
    print(f"  visible in frame: {n_visible:,} / {len(points):,} ({100 * n_visible / len(points):.1f}%)")
    if n_visible == 0:
        raise RuntimeError(f"No visible points in frame {FRAME_IDX}")

    frame_feats = load_or_extract_frame_features(out_dir)
    expected = (cfg.PATCH_GRID_H, cfg.PATCH_GRID_W, cfg.DINOTXT_FEATURE_DIM)
    if frame_feats.shape != expected:
        raise ValueError(f"Expected feature shape {expected}, got {frame_feats.shape}")

    px = frame_vis["patch_x"].astype(np.int64)
    py = frame_vis["patch_y"].astype(np.int64)
    vis_feats = frame_feats[py, px]

    print(f"PCA-coloring {n_visible:,} visible vertex features...")
    vis_colors = pca_features_to_rgb(vis_feats)
    all_colors = build_mesh_colors(len(points), vis_idx, vis_colors)

    tag = f"frame_{FRAME_IDX:06d}"
    ply_all = out_dir / f"dinotxt_pca_{tag}_all_vertices.ply"
    ply_vis = out_dir / f"dinotxt_pca_{tag}_visible.ply"
    scatter_all = out_dir / f"dinotxt_pca_{tag}_all_vertices.png"
    scatter_vis = out_dir / f"dinotxt_pca_{tag}_visible.png"

    save_colored_ply(points, all_colors, ply_all)
    save_colored_ply(points[vis_idx], vis_colors, ply_vis)
    save_scatter_png(
        points,
        all_colors,
        scatter_all,
        f"DINOtxt PCA — frame {FRAME_IDX} (all {len(points):,} vertices, gray = not visible)",
    )
    save_scatter_png(
        points[vis_idx],
        vis_colors,
        scatter_vis,
        f"DINOtxt PCA — {n_visible:,} visible vertices",
    )

    # 2D patch-grid PCA (all patches in the image)
    pca_patch = pca_patch_grid_to_rgb(frame_feats)
    pca_full = upsample_patch_image(pca_patch)
    pca_patch_path = out_dir / f"dinotxt_pca_{tag}_patchgrid.png"
    pca_full_path = out_dir / f"dinotxt_pca_{tag}_2d.png"
    compare_path = out_dir / f"dinotxt_pca_{tag}_comparison.png"
    Image.fromarray(pca_patch).save(pca_patch_path)
    Image.fromarray(pca_full).save(pca_full_path)
    rgb = load_rgb(FRAME_IDX)
    save_comparison_figure(rgb, pca_full, compare_path, FRAME_IDX)

    norms = np.linalg.norm(vis_feats, axis=1)
    summary = {
        "frame_idx": FRAME_IDX,
        "num_mesh_vertices": int(len(points)),
        "num_visible": int(n_visible),
        "visible_fraction": float(n_visible / len(points)),
        "feature_shape": list(frame_feats.shape),
        "feature_dim": cfg.DINOTXT_FEATURE_DIM,
        "mean_visible_patch_norm": float(norms.mean()),
        "rgb_source": str(frame_path(FRAME_IDX)),
        "legend": "PCA RGB on visible vertices; gray = not visible in frame",
        "outputs": {
            "all_vertices_ply": str(ply_all),
            "visible_ply": str(ply_vis),
            "all_vertices_scatter": str(scatter_all),
            "visible_scatter": str(scatter_vis),
            "pca_2d_fullres": str(pca_full_path),
            "pca_2d_patchgrid": str(pca_patch_path),
            "comparison": str(compare_path),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  {ply_all.name}   — all {len(points):,} mesh vertices (PCA + gray)")
    print(f"  {ply_vis.name} — {n_visible:,} visible vertices only")
    print(f"  {pca_full_path.name}              — 2D patch-grid PCA")
    print(f"  mean visible patch L2 norm: {norms.mean():.4f}")

    if SHOW_VIEWER:
        show_open3d(ply_all, title=f"DINOtxt PCA — all vertices (frame {FRAME_IDX})")


if __name__ == "__main__":
    main()
