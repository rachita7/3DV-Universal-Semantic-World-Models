#!/usr/bin/env python3
"""
DINOtxt text–vision similarity test on full mesh vertices for one frame.

Encodes a class prompt (e.g. chair) with the DINOtxt text encoder, extracts
DINOtxt aligned patch features for visible mesh vertices in a single frame,
computes cosine similarity, and saves a 3D heatmap PLY + scatter PNG.

Run from repo root:
    python test/test_dinotxt_similarity.py

Edit TARGET_CLASS, FRAME_IDX, and SHOW_VIEWER below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- config (edit these) ---
TARGET_CLASS = "chair" # must exist in config.settings.TEXT_PROMPTS
FRAME_IDX = 1
DEVICE = "cuda"
SHOW_VIEWER = True  # set False on headless/SSH machines
FORCE_RECOMPUTE = True  # required after text/patch embedding fix; safe to set False once re-run
# ---------------------------

COLOR_NOT_VISIBLE = (80, 80, 80)

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_DIR = TEST_DIR / "test_outputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from config import settings as cfg
from src.data.mesh import load_semantic_mesh
from src.data.projection import STATUS_VISIBLE, project_frame_visibility
from src.data.replica import frame_path, load_camera_params, load_rgb, load_trajectory
from src.features.dinotxt_features import extract_aligned_patches
from src.features.dinotxt_text import (
    cosine_similarity_to_text,
    ensemble_class_embedding,
    load_dinotxt,
)
from src.features.preprocess import rgb_to_tensor
from src.utils.io import load_npy, save_npy
from src.viz.gt_labels import save_colored_ply


def run_dir_name() -> str:
    return f"dinotxt_sim_{TARGET_CLASS}_frame_{FRAME_IDX:06d}"


def similarity_to_jet_colors(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    cmap = plt.colormaps["jet"]
    v = values.astype(np.float64)
    if vmin is None:
        vmin = float(np.percentile(v, 2)) if len(v) else 0.0
    if vmax is None:
        vmax = float(np.percentile(v, 98)) if len(v) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-8
    norm = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
    return (cmap(norm)[:, :3] * 255).astype(np.uint8)


def build_full_mesh_colors(
    n_points: int,
    visible_indices: np.ndarray,
    visible_similarities: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    colors = np.tile(np.array(COLOR_NOT_VISIBLE, dtype=np.uint8), (n_points, 1))
    jet = similarity_to_jet_colors(visible_similarities, vmin=vmin, vmax=vmax)
    colors[visible_indices] = jet
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


def save_histogram(values: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=80, color="steelblue", edgecolor="white")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def show_open3d(ply_path: Path, title: str) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — skip viewer")
        return
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"  Open3D: {title} ({len(pcd.points)} points) — close window to exit")
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=title,
            width=1280,
            height=720,
        )
    except Exception as exc:
        print(f"  Viewer failed ({exc}). Open PLY manually: {ply_path}")


def encode_class_embedding(class_name: str, device: str) -> np.ndarray:
    if class_name not in cfg.TEXT_PROMPTS:
        available = ", ".join(sorted(cfg.TEXT_PROMPTS))
        raise ValueError(f"Unknown class {class_name!r}. Available: {available}")
    prompts = cfg.TEXT_PROMPTS[class_name]
    model, tokenizer = load_dinotxt(device)
    emb = ensemble_class_embedding(model, tokenizer, prompts, device)
    print(f"  encoded '{class_name}' from {len(prompts)} prompts: shape={emb.shape}")
    return emb


def extract_frame_features(frame_idx: int, device: str, cache_path: Path, force: bool) -> np.ndarray:
    if cache_path.exists() and not force:
        print(f"Using cached DINOtxt frame features: {cache_path.name}")
        return load_npy(cache_path)

    print(f"Extracting DINOtxt features for frame {frame_idx} ({device})...")
    model, _ = load_dinotxt(device)
    rgb = load_rgb(frame_idx)
    tensor = rgb_to_tensor(rgb).to(device)
    feat = extract_aligned_patches(model, tensor)
    save_npy(cache_path, feat)
    print(f"  saved frame features {feat.shape} -> {cache_path.name}")
    return feat


def main() -> None:
    out_dir = OUTPUT_DIR / run_dir_name()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = DEVICE if DEVICE != "cuda" or torch.cuda.is_available() else "cpu"
    if TARGET_CLASS not in cfg.TEXT_PROMPTS:
        raise ValueError(f"TARGET_CLASS={TARGET_CLASS!r} not in cfg.TEXT_PROMPTS")

    poses = load_trajectory()
    if FRAME_IDX < 0 or FRAME_IDX >= len(poses):
        raise ValueError(f"FRAME_IDX={FRAME_IDX} out of range [0, {len(poses) - 1}]")

    # --- load full mesh ---
    print(f"Loading all vertices from {cfg.MESH_PATH}...")
    vertices, _, _ = load_semantic_mesh(cfg.MESH_PATH)
    points = vertices.astype(np.float32)
    print(f"  {len(points):,} mesh vertices")

    # --- text embedding ---
    text_emb_path = out_dir / f"text_embedding_{TARGET_CLASS}.npy"
    if text_emb_path.exists() and not FORCE_RECOMPUTE:
        text_emb = load_npy(text_emb_path)
        print(f"Loaded text embedding from {text_emb_path.name}")
    else:
        print(f"Encoding text prompts for '{TARGET_CLASS}'...")
        text_emb = encode_class_embedding(TARGET_CLASS, device)
        save_npy(text_emb_path, text_emb)

    # --- single-frame visibility ---
    cam = load_camera_params()
    print(f"Projecting into frame {FRAME_IDX} ({frame_path(FRAME_IDX).name})...")
    frame_vis = project_frame_visibility(points, FRAME_IDX, poses, cam)
    vis_idx = frame_vis["visible_indices"]
    n_visible = len(vis_idx)
    print(f"  visible: {n_visible:,} / {len(points):,} ({100 * n_visible / len(points):.1f}%)")

    if n_visible == 0:
        raise RuntimeError(f"No visible points in frame {FRAME_IDX}")

    # --- DINOtxt patch features for this frame ---
    feat_cache = out_dir / f"features_dinotxt_{FRAME_IDX:06d}.npy"
    frame_feats = extract_frame_features(FRAME_IDX, device, feat_cache, force=FORCE_RECOMPUTE)

    px = frame_vis["patch_x"].astype(np.int64)
    py = frame_vis["patch_y"].astype(np.int64)
    point_feats = frame_feats[py, px].astype(np.float32)

    feat_norms = np.linalg.norm(point_feats, axis=1)
    text_norm = float(np.linalg.norm(text_emb))
    print(f"  patch feature L2 norm: mean={feat_norms.mean():.4f}, max={feat_norms.max():.4f}")
    print(f"  text embedding L2 norm: {text_norm:.4f}")

    similarities = cosine_similarity_to_text(point_feats, text_emb)
    sim_min = float(similarities.min())
    sim_max = float(similarities.max())
    sim_mean = float(similarities.mean())
    print(f"\nCosine similarity to '{TARGET_CLASS}' (visible points only):")
    print(f"  mean = {sim_mean:.4f}, min = {sim_min:.4f}, max = {sim_max:.4f}")
    if sim_max > 1.0 + 1e-3 or sim_min < -1.0 - 1e-3:
        raise RuntimeError(
            "Similarity out of [-1, 1] — patch/text vectors may not be L2-normalized "
            "or text embedding uses the wrong 2048-d half. Re-run with FORCE_RECOMPUTE=True."
        )

    # Jet scale from visible similarities (2–98 percentile)
    sim_vmin = float(np.percentile(similarities, 2))
    sim_vmax = float(np.percentile(similarities, 98))

    # --- visualize ---
    full_colors = build_full_mesh_colors(len(points), vis_idx, similarities, sim_vmin, sim_vmax)
    vis_colors = similarity_to_jet_colors(similarities, vmin=sim_vmin, vmax=sim_vmax)

    ply_all = out_dir / f"similarity_{TARGET_CLASS}_all_vertices.ply"
    ply_vis = out_dir / f"similarity_{TARGET_CLASS}_visible.ply"
    png_all = out_dir / f"similarity_{TARGET_CLASS}_all_vertices.png"
    png_vis = out_dir / f"similarity_{TARGET_CLASS}_visible.png"
    hist_path = out_dir / f"similarity_{TARGET_CLASS}_histogram.png"

    save_colored_ply(points, full_colors, ply_all)
    save_colored_ply(points[vis_idx], vis_colors, ply_vis)
    save_scatter_png(
        points,
        full_colors,
        png_all,
        f"'{TARGET_CLASS}' cosine similarity — frame {FRAME_IDX} (gray = not visible)",
    )
    save_scatter_png(
        points[vis_idx],
        vis_colors,
        png_vis,
        f"'{TARGET_CLASS}' cosine similarity — {n_visible:,} visible vertices",
    )
    save_histogram(
        similarities,
        hist_path,
        f"Cosine similarity to '{TARGET_CLASS}' (frame {FRAME_IDX}, visible only)",
    )

    sim_full = np.full(len(points), np.nan, dtype=np.float32)
    sim_full[vis_idx] = similarities
    save_npy(out_dir / f"similarity_{TARGET_CLASS}.npy", sim_full)
    save_npy(out_dir / "visible_indices.npy", vis_idx.astype(np.int32))

    summary = {
        "target_class": TARGET_CLASS,
        "frame_idx": FRAME_IDX,
        "num_mesh_vertices": int(len(points)),
        "num_visible": int(n_visible),
        "visible_fraction": float(n_visible / len(points)),
        "similarity_mean_visible": sim_mean,
        "similarity_min_visible": sim_min,
        "similarity_max_visible": sim_max,
        "jet_scale": [sim_vmin, sim_vmax],
        "text_prompts": cfg.TEXT_PROMPTS[TARGET_CLASS],
        "rgb_frame": str(frame_path(FRAME_IDX)),
        "legend": "jet colormap on visible vertices; gray = not visible in frame",
        "outputs": {
            "text_embedding": str(text_emb_path),
            "frame_features": str(feat_cache),
            "similarity_npy": str(out_dir / f"similarity_{TARGET_CLASS}.npy"),
            "all_vertices_ply": str(ply_all),
            "visible_ply": str(ply_vis),
            "all_vertices_png": str(png_all),
            "visible_png": str(png_vis),
            "histogram_png": str(hist_path),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  {ply_all.name}   — full mesh (jet + gray)")
    print(f"  {ply_vis.name} — visible-only heatmap")
    print(f"  {png_all.name}")
    print(f"  {hist_path.name}")

    if SHOW_VIEWER:
        show_open3d(ply_all, title=f"'{TARGET_CLASS}' similarity — all vertices")
        show_open3d(ply_vis, title=f"'{TARGET_CLASS}' similarity — visible only")


if __name__ == "__main__":
    main()
