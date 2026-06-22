"""Cosine-similarity segmentation of 3D points."""

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
from src.features.dinotxt_text import load_class_embeddings
from src.utils.io import ensure_dir, load_npy


def results_dir() -> Path:
    return ensure_dir(output_dir() / "results")


def classify_points(
    point_features: np.ndarray,
    class_names: list[str],
    class_embeddings: np.ndarray,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify points by cosine similarity to class prototypes.

    Returns:
        pred_class_indices (N,)
        pred_scores (N,) max cosine similarity
        pred_class_names (N,) object
    """
    threshold = threshold if threshold is not None else cfg.COSINE_THRESHOLD
    # Both already normalized
    sims = point_features @ class_embeddings.T  # (N, C)
    pred_idx = sims.argmax(axis=1)
    pred_scores = sims[np.arange(len(point_features)), pred_idx]

    if threshold > 0:
        pred_idx = np.where(pred_scores >= threshold, pred_idx, -1)

    pred_names = np.array(
        [class_names[i] if i >= 0 else "unknown" for i in pred_idx],
        dtype=object,
    )
    return pred_idx.astype(np.int32), pred_scores.astype(np.float32), pred_names


def run_segmentation(
    aggregated_path: Path,
    subsample_idx_path: Path | None = None,
    force: bool = False,
) -> dict:
    """Step 9: segment points and save predictions."""
    method = aggregated_path.stem
    out_pred = results_dir() / f"predictions_{method}.npy"
    out_pred_names = results_dir() / f"predictions_{method}_names.npy"

    if out_pred.exists() and not force:
        pred_names = np.load(out_pred_names, allow_pickle=True)
        return {"pred_names": pred_names, "method": method}

    point_features = load_npy(aggregated_path)
    class_names, class_embeddings = load_class_embeddings()

    pred_idx, pred_scores, pred_names = classify_points(
        point_features, class_names, class_embeddings
    )

    np.save(out_pred, pred_idx)
    np.save(out_pred_names, pred_names)
    np.save(results_dir() / f"pred_scores_{method}.npy", pred_scores)

    print(f"Segmented {len(point_features)} points with method={method}")
    return {
        "pred_idx": pred_idx,
        "pred_scores": pred_scores,
        "pred_names": pred_names,
        "class_names": class_names,
        "method": method,
    }


def main():
    agg_path = output_dir() / "aggregated" / "mean.npy"
    if not agg_path.exists():
        print("No aggregated features found; run pipeline first.")
        return
    result = run_segmentation(agg_path, force=True)
    print(f"Predictions: {result['pred_names'][:5]}")


if __name__ == "__main__":
    main()
