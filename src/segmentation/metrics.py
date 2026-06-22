"""IoU and mIoU metrics for 3D semantic segmentation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import numpy as np

from config import settings as cfg
from src.data.mesh import gt_labels_path, load_info_semantic, object_ids_to_class_names, output_dir
from src.segmentation.classify import results_dir
from src.utils.io import ensure_dir, load_npy


def compute_iou_per_class(
    gt_names: np.ndarray,
    pred_names: np.ndarray,
    classes: list[str],
) -> dict[str, float]:
    ious = {}
    for cls in classes:
        gt_mask = gt_names == cls
        pred_mask = pred_names == cls
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        ious[cls] = float(intersection / union) if union > 0 else float("nan")
    return ious


def compute_miou(ious: dict[str, float]) -> float:
    valid = [v for v in ious.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def evaluate_segmentation(
    pred_names: np.ndarray,
    gt_object_ids: np.ndarray,
    eval_classes: list[str] | None = None,
) -> dict:
    obj_to_class, _ = load_info_semantic(cfg.INFO_SEMANTIC_PATH)
    gt_names = object_ids_to_class_names(gt_object_ids, obj_to_class)

    if eval_classes is None:
        eval_classes = cfg.EVAL_CLASSES
    if eval_classes is None:
        # classes present in GT that also appear in predictions
        unique_gt = set(gt_names)
        unique_pred = set(pred_names)
        eval_classes = sorted(unique_gt & unique_pred - {"unknown"})

    ious = compute_iou_per_class(gt_names, pred_names, eval_classes)
    miou = compute_miou(ious)

    # accuracy on labeled points
    valid = pred_names != "unknown"
    accuracy = float((gt_names[valid] == pred_names[valid]).mean()) if valid.any() else 0.0

    return {
        "miou": miou,
        "accuracy": accuracy,
        "per_class_iou": ious,
        "num_points": len(gt_names),
        "eval_classes": eval_classes,
    }


def save_metrics(metrics: dict, method: str) -> Path:
    out = results_dir() / f"metrics_{method}.json"
    ensure_dir(out.parent)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"mIoU={metrics['miou']:.4f}, accuracy={metrics['accuracy']:.4f} -> {out}")
    return out


def run_evaluation(method: str, subsample_idx: np.ndarray | None = None) -> dict:
    """Evaluate predictions against GT for subsampled points."""
    pred_names = np.load(results_dir() / f"predictions_{method}_names.npy", allow_pickle=True)
    gt_all = load_npy(gt_labels_path(cfg.NUM_POINTS_INITIAL))

    if subsample_idx is not None:
        gt_object_ids = gt_all[subsample_idx]
    else:
        idx_path = output_dir() / f"points_{cfg.NUM_POINTS_SUBSAMPLE}_idx.npy"
        if idx_path.exists():
            subsample_idx = load_npy(idx_path)
            gt_object_ids = gt_all[subsample_idx]
        else:
            gt_object_ids = load_npy(gt_labels_path(cfg.NUM_POINTS_SUBSAMPLE))

    metrics = evaluate_segmentation(pred_names, gt_object_ids)
    save_metrics(metrics, method)
    return metrics


def run_ablation(subsample_idx: np.ndarray | None = None) -> dict:
    """Phase 4: compare all methods in ABLATION_METHODS."""
    summary = {}
    for method in cfg.ABLATION_METHODS:
        metrics_path = results_dir() / f"metrics_{method}.json"
        if not metrics_path.exists():
            print(f"Skipping {method}: no metrics found")
            continue
        with open(metrics_path) as f:
            summary[method] = json.load(f)

    out = results_dir() / "ablation_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Ablation summary saved to {out}")
    return summary


def main():
    print("Metrics module — run via main.py step 9 or run_ablation()")


if __name__ == "__main__":
    main()
