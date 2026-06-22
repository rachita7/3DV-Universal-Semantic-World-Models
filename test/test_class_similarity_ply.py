#!/usr/bin/env python3
"""
Save a PLY heatmap of cosine similarity to one text class over the 3D scene.

Run from repo root:
    conda run -n 3dvision python test/test_class_similarity_ply.py

Edit TARGET_CLASS and AGGREGATION_METHOD below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- config (edit these) ---
TARGET_CLASS = "window"
AGGREGATION_METHOD = "sam_slerp"  # "mean" | "frechet" | "weighted_slerp" | "sam_slerp"
FORCE_RECOMPUTE_TEXT = False
SHOW_VIEWER = True
# ---------------------------

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import settings as cfg
from src.data.mesh import output_dir
from src.features.dinotxt_text import (
    build_class_embeddings,
    cosine_similarity_to_text,
    text_embeddings_dir,
)
from src.utils.io import load_npy, save_npy
from src.viz.gt_labels import save_colored_ply
from src.viz.ply_viewer import ISO_FRONT, ISO_UP, ISO_ZOOM


def class_similarity_dir() -> Path:
    out_dir = output_dir() / "viz" / "class_similarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def similarity_to_jet_colors(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        vmax = vmin + 1e-8

    norm = np.clip((values.astype(np.float64) - vmin) / (vmax - vmin), 0.0, 1.0)
    colors = plt.colormaps["jet"](norm)[:, :3]
    return (colors * 255).astype(np.uint8), (vmin, vmax)


def load_text_embedding(class_name: str) -> np.ndarray:
    if class_name not in cfg.TEXT_PROMPTS:
        available = ", ".join(sorted(cfg.TEXT_PROMPTS))
        raise ValueError(f"Unknown TARGET_CLASS={class_name!r}. Available classes: {available}")

    emb_path = text_embeddings_dir() / f"{class_name}.npy"
    if emb_path.exists() and not FORCE_RECOMPUTE_TEXT:
        return load_npy(emb_path).astype(np.float32)

    embeddings = build_class_embeddings(
        class_names=[class_name],
        force=FORCE_RECOMPUTE_TEXT,
    )
    return embeddings[class_name].astype(np.float32)


def show_ply(ply_path: Path, title: str) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("open3d is not installed; skipping viewer.")
        return

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        print(f"PLY is empty; skipping viewer: {ply_path}")
        return

    points = np.asarray(pcd.points)
    print(f"Opening viewer: {ply_path} ({len(points):,} points)")
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=title,
            width=1280,
            height=720,
            lookat=points.mean(axis=0),
            front=ISO_FRONT,
            up=ISO_UP,
            zoom=ISO_ZOOM,
        )
    except Exception as exc:
        print(f"Viewer failed ({exc}). Open PLY manually: {ply_path}")


def main() -> None:
    root = output_dir()
    points_path = root / f"points_{cfg.NUM_POINTS_SUBSAMPLE}.npy"
    agg_path = root / "aggregated" / f"{AGGREGATION_METHOD}.npy"

    if not points_path.exists():
        raise FileNotFoundError(f"Missing subsampled points: {points_path}")
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing aggregated features: {agg_path}")

    points = load_npy(points_path).astype(np.float32)
    point_features = load_npy(agg_path).astype(np.float32)
    if len(points) != len(point_features):
        raise ValueError(
            f"Point/features length mismatch: {len(points)} points vs "
            f"{len(point_features)} feature rows"
        )

    text_embedding = load_text_embedding(TARGET_CLASS)
    similarities = cosine_similarity_to_text(point_features, text_embedding)
    colors, scale = similarity_to_jet_colors(similarities)

    out_dir = class_similarity_dir()
    stem = f"similarity_{TARGET_CLASS}_{AGGREGATION_METHOD}"
    ply_path = out_dir / f"{stem}.ply"
    sim_path = out_dir / f"{stem}.npy"
    summary_path = out_dir / f"{stem}_summary.json"

    save_colored_ply(points, colors, ply_path)
    save_npy(sim_path, similarities)
    summary = {
        "target_class": TARGET_CLASS,
        "aggregation_method": AGGREGATION_METHOD,
        "num_points": int(len(points)),
        "similarity_min": float(similarities.min()),
        "similarity_max": float(similarities.max()),
        "similarity_mean": float(similarities.mean()),
        "jet_scale_type": "min_max_cosine_similarity",
        "jet_value_range": list(scale),
        "points_path": str(points_path),
        "aggregated_path": str(agg_path),
        "outputs": {
            "ply": str(ply_path),
            "similarity_npy": str(sim_path),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved class similarity PLY: {ply_path}")
    print(f"Saved raw similarities: {sim_path}")
    print(
        f"{TARGET_CLASS!r} similarity ({AGGREGATION_METHOD}): "
        f"min={summary['similarity_min']:.4f}, "
        f"mean={summary['similarity_mean']:.4f}, "
        f"max={summary['similarity_max']:.4f}"
    )
    print(f"Jet scale: {scale[0]:.4f} to {scale[1]:.4f}")

    if SHOW_VIEWER:
        show_ply(
            ply_path,
            title=f"{TARGET_CLASS} cosine similarity ({AGGREGATION_METHOD})",
        )


if __name__ == "__main__":
    main()
