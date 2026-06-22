"""Visualize ground-truth labels as a colored point cloud."""

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
    points_path,
)
from src.utils.io import ensure_dir, load_npy

# BGR-style RGB tuples for PLY
COLOR_CHAIR = (0, 200, 0)
COLOR_TABLE = (0, 100, 255)
COLOR_OTHER = (160, 160, 160)

# Optional palette for multi-class PLY (class_name -> RGB)
CLASS_PALETTE: dict[str, tuple[int, int, int]] = {
    "chair": COLOR_CHAIR,
    "table": COLOR_TABLE,
    "sofa": (255, 128, 0),
    "bed": (180, 0, 180),
    "wall": (200, 200, 100),
    "floor": (139, 90, 43),
    "ceiling": (220, 220, 220),
    "rug": (255, 0, 128),
    "lamp": (255, 255, 0),
    "plant": (0, 180, 0),
}


def viz_dir() -> Path:
    return ensure_dir(output_dir() / "viz")


def class_to_color(class_name: str, highlight: set[str] | None = None) -> tuple[int, int, int]:
    if highlight is not None:
        if class_name == "chair":
            return COLOR_CHAIR
        if class_name == "table":
            return COLOR_TABLE
        if class_name not in highlight:
            return COLOR_OTHER
    return CLASS_PALETTE.get(class_name, COLOR_OTHER)


def save_colored_ply(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
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
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def run_gt_viz(
    num_points: int | None = None,
    highlight_classes: set[str] | None = None,
    force: bool = False,
) -> tuple[Path, Path | None]:
    """
    Color sampled points by GT class and save PLY files.

    Returns paths to chair/table highlight PLY and optional multi-class PLY.
    """
    num_points = num_points if num_points is not None else cfg.NUM_POINTS_INITIAL
    highlight_classes = highlight_classes if highlight_classes is not None else {"chair", "table"}

    out_chair_table = viz_dir() / "gt_chair_table.ply"
    out_multiclass = viz_dir() / "gt_multiclass.ply"

    if out_chair_table.exists() and not force:
        print(f"GT viz exists at {out_chair_table}")
        return out_chair_table, out_multiclass if out_multiclass.exists() else None

    points = load_npy(points_path(num_points))
    gt_ids = load_npy(gt_labels_path(num_points))
    obj_to_class, _ = load_info_semantic(cfg.INFO_SEMANTIC_PATH)
    class_names = object_ids_to_class_names(gt_ids, obj_to_class)

    colors_hl = np.array(
        [class_to_color(cn, highlight=highlight_classes) for cn in class_names],
        dtype=np.uint8,
    )
    save_colored_ply(points, colors_hl, out_chair_table)

    colors_mc = np.array(
        [class_to_color(cn, highlight=None) for cn in class_names],
        dtype=np.uint8,
    )
    save_colored_ply(points, colors_mc, out_multiclass)

    n_chair = int((class_names == "chair").sum())
    n_table = int((class_names == "table").sum())
    print(f"Saved GT viz: {out_chair_table} (chair={n_chair}, table={n_table}, total={len(points)})")
    print(f"Saved multi-class viz: {out_multiclass}")
    return out_chair_table, out_multiclass


def main():
    run_gt_viz(force=True)
    obj_to_class, _ = load_info_semantic(cfg.INFO_SEMANTIC_PATH)
    gt_ids = load_npy(gt_labels_path(cfg.NUM_POINTS_INITIAL))
    names = object_ids_to_class_names(gt_ids, obj_to_class)
    unique, counts = np.unique(names, return_counts=True)
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"  {u}: {c}")


if __name__ == "__main__":
    main()
