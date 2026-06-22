"""Visualize segmentation errors: TP / FP / FN colored point cloud."""

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

from config import settings as cfg
from src.data.mesh import (
    gt_labels_path,
    load_info_semantic,
    object_ids_to_class_names,
    output_dir,
)
from src.segmentation.classify import results_dir
from src.utils.io import load_npy


# Colors: TP=green, FP=red, FN=yellow, TN=gray (for background-like classes)
COLOR_TP = (0, 200, 0)
COLOR_FP = (220, 0, 0)
COLOR_FN = (255, 200, 0)
COLOR_UNKNOWN = (128, 128, 128)


def error_colors(gt_names: np.ndarray, pred_names: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(gt_names), 3), dtype=np.uint8)
    for i in range(len(gt_names)):
        gt, pred = gt_names[i], pred_names[i]
        if pred == "unknown":
            colors[i] = COLOR_UNKNOWN
        elif gt == pred:
            colors[i] = COLOR_TP
        elif pred != gt:
            # wrong class assigned
            colors[i] = COLOR_FP
    # FN: GT has class but predicted wrong — already FP from pred side
    # Explicit FN marker: GT non-unknown, pred different
    fn_mask = (gt_names != pred_names) & (pred_names != "unknown")
    colors[fn_mask] = COLOR_FP
    unknown_gt_wrong = (gt_names != pred_names) & (pred_names == "unknown")
    colors[unknown_gt_wrong] = COLOR_FN
    return colors


def save_error_ply(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def run_segmentation_viz(method: str, force: bool = False) -> Path:
    """Step 9 viz: error-colored point cloud."""
    out_path = results_dir() / f"segmentation_errors_{method}.ply"
    if out_path.exists() and not force:
        return out_path

    points = load_npy(output_dir() / f"points_{cfg.NUM_POINTS_SUBSAMPLE}.npy")
    pred_names = np.load(results_dir() / f"predictions_{method}_names.npy", allow_pickle=True)

    idx_path = output_dir() / f"points_{cfg.NUM_POINTS_SUBSAMPLE}_idx.npy"
    subsample_idx = load_npy(idx_path)
    gt_all = load_npy(gt_labels_path(cfg.NUM_POINTS_INITIAL))
    gt_object_ids = gt_all[subsample_idx]

    obj_to_class, _ = load_info_semantic(cfg.INFO_SEMANTIC_PATH)
    gt_names = object_ids_to_class_names(gt_object_ids, obj_to_class)

    colors = error_colors(gt_names, pred_names)
    save_error_ply(points, colors, out_path)
    n_tp = np.all(colors == COLOR_TP, axis=1).sum()
    n_wrong = len(points) - n_tp - (colors == COLOR_UNKNOWN).all(axis=1).sum()
    print(f"Saved error viz to {out_path} (approx correct={n_tp}/{len(points)})")
    return out_path


def main():
    method = cfg.AGGREGATION_METHOD
    pred_path = results_dir() / f"predictions_{method}_names.npy"
    if not pred_path.exists():
        print("No predictions found; run pipeline first.")
        return
    run_segmentation_viz(method, force=True)


if __name__ == "__main__":
    main()
