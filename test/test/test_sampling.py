#!/usr/bin/env python3
"""
Step 1 sampling test — sample mesh vertices and visualize GT labels.

Run from repo root:
    python test/test_sampling.py

Edit NUM_POINTS below and re-run to try different sample sizes.
All artifacts go to test/test_outputs/.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- config (edit these) ---
NUM_POINTS = 5_000_000
RANDOM_SEED = 42
SHOW_VIEWER = True  # Open3D window; set False on headless machines
# ---------------------------

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_DIR = TEST_DIR / "test_outputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import settings as cfg
from src.data.mesh import load_info_semantic, object_ids_to_class_names, sample_mesh_points
from src.viz.gt_labels import class_to_color, save_colored_ply


def save_points_npy(points: np.ndarray, gt_ids: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"points_{len(points)}.npy", points)
    np.save(out_dir / f"gt_labels_{len(points)}.npy", gt_ids)


def build_colors(class_names: np.ndarray, multiclass: bool = True) -> np.ndarray:
    if multiclass:
        return np.array(
            [class_to_color(cn, highlight=None) for cn in class_names],
            dtype=np.uint8,
        )
    return np.array(
        [class_to_color(cn, highlight={"chair", "table"}) for cn in class_names],
        dtype=np.uint8,
    )


def save_class_histogram(class_names: np.ndarray, out_path: Path, top_k: int = 15) -> None:
    counts = Counter(class_names)
    items = counts.most_common(top_k)
    labels, values = zip(*items) if items else ([], [])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(list(reversed(labels)), list(reversed(values)), color="steelblue")
    ax.set_xlabel("Point count")
    ax.set_title(f"GT class distribution ({len(class_names)} sampled points)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_scatter_png(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    """Lightweight 3D preview (works without a display)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    rgb = colors.astype(np.float32) / 255.0
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=rgb,
        s=0.5,
        alpha=0.8,
        marker=".",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Sampled points (N={len(points)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def show_open3d(ply_path: Path) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — skip interactive viewer")
        return

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        return
    print(f"  Open3D viewer: {ply_path.name} ({len(pcd.points)} points) — close window to exit")
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=ply_path.name,
            width=1280,
            height=720,
        )
    except Exception as exc:
        print(f"  Could not open viewer ({exc}). Use the PNG/PLY in test_outputs/")


def main() -> None:
    run_dir = OUTPUT_DIR / f"sample_{NUM_POINTS}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {NUM_POINTS} vertices (seed={RANDOM_SEED})...")
    print(f"Mesh: {cfg.MESH_PATH}")
    points, _vertex_indices, gt_object_ids = sample_mesh_points(NUM_POINTS, RANDOM_SEED)

    obj_to_class, _ = load_info_semantic(cfg.INFO_SEMANTIC_PATH)
    class_names = object_ids_to_class_names(gt_object_ids, obj_to_class)

    save_points_npy(points, gt_object_ids, run_dir)

    colors_mc = build_colors(class_names, multiclass=True)
    colors_hl = build_colors(class_names, multiclass=False)
    ply_mc = run_dir / "sample_multiclass.ply"
    ply_hl = run_dir / "sample_chair_table.ply"
    save_colored_ply(points, colors_mc, ply_mc)
    save_colored_ply(points, colors_hl, ply_hl)

    save_class_histogram(class_names, run_dir / "class_histogram.png")
    save_scatter_png(points, colors_mc, run_dir / "sample_scatter.png")

    unique_ids, _id_counts = np.unique(gt_object_ids, return_counts=True)
    summary = {
        "num_points": int(len(points)),
        "random_seed": RANDOM_SEED,
        "mesh_path": cfg.MESH_PATH,
        "unique_object_ids": int(len(unique_ids)),
        "unique_class_names": int(len(set(class_names))),
        "top_classes": dict(Counter(class_names).most_common(10)),
        "outputs": {
            "points": str(run_dir / f"points_{len(points)}.npy"),
            "gt_labels": str(run_dir / f"gt_labels_{len(points)}.npy"),
            "ply_multiclass": str(ply_mc),
            "ply_chair_table": str(ply_hl),
            "scatter_png": str(run_dir / "sample_scatter.png"),
            "histogram_png": str(run_dir / "class_histogram.png"),
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(points)} points to {run_dir}/")
    print(f"  PLY (multiclass):  {ply_mc.name}")
    print(f"  PLY (chair/table): {ply_hl.name}")
    print(f"  PNG scatter:       sample_scatter.png")
    print(f"  PNG histogram:     class_histogram.png")
    print(f"  Top classes:       {list(summary['top_classes'].items())[:5]}")

    if SHOW_VIEWER:
        show_open3d(ply_mc)


if __name__ == "__main__":
    main()
