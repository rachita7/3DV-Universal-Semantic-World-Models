#!/usr/bin/env python3
"""
Semantic flickering research pipeline.

Edit config/settings.py to change paths, parameters, and which steps to run.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings as cfg
from src.aggregation.slerp_agg import aggregate_all_points as slerp_aggregate_all
from src.aggregation.variance import compute_all_dispersion
from src.data.mesh import (
    gt_labels_path,
    output_dir,
    points_path,
    run_sample_points,
    run_subsample_points,
)
from src.data.projection import run_build_visibility
from src.features.dino_backbone import extract_all_frames as extract_vits16
from src.features.dinotxt_features import extract_all_frames as extract_dinotxt
from src.features.dinotxt_text import build_class_embeddings
from src.segmentation.classify import run_segmentation
from src.segmentation.metrics import run_ablation, run_evaluation
from src.utils.io import save_settings_snapshot
from src.viz.segmentation_viz import run_segmentation_viz
from src.viz.variance_heatmap import run_variance_viz


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_step(step: int, force: bool) -> None:
    print(f"\n{'=' * 60}\nStep {step}\n{'=' * 60}")

    if step == 1:
        run_sample_points(force=force)
        from src.viz.gt_labels import run_gt_viz

        run_gt_viz(force=force)

    elif step == 2:
        run_build_visibility(force=force)

    elif step == 3:
        extract_vits16(force=force)

    elif step == 4:
        compute_all_dispersion(force=force)
        run_variance_viz(force=force)

    elif step == 5:
        from src.aggregation.variance import dispersion_path

        run_subsample_points(dispersion_path(), force=force)

    elif step == 6:
        extract_dinotxt(force=force)

    elif step == 7:
        idx_path = output_dir() / f"points_{cfg.NUM_POINTS_SUBSAMPLE}_idx.npy"
        subsample_idx = np.load(idx_path) if idx_path.exists() else None
        n_points = cfg.NUM_POINTS_SUBSAMPLE if subsample_idx is not None else cfg.NUM_POINTS_INITIAL
        method = cfg.AGGREGATION_METHOD
        slerp_aggregate_all(
            n_points=n_points,
            point_indices=subsample_idx,
            method=method,
            force=force,
        )

    elif step == 8:
        build_class_embeddings(force=force)

    elif step == 9:
        method = cfg.AGGREGATION_METHOD
        agg_path = output_dir() / "aggregated" / f"{method}.npy"
        if not agg_path.exists():
            raise FileNotFoundError(f"Aggregated features not found: {agg_path}. Run step 7 first.")

        idx_path = output_dir() / f"points_{cfg.NUM_POINTS_SUBSAMPLE}_idx.npy"
        subsample_idx = np.load(idx_path) if idx_path.exists() else None

        run_segmentation(agg_path, force=force)
        run_evaluation(method, subsample_idx=subsample_idx)
        run_segmentation_viz(method, force=force)

    else:
        raise ValueError(f"Unknown step: {step}")


def run_ablation_pipeline(force: bool = False) -> None:
    """Phase 4: run aggregation + segmentation for each method in ABLATION_METHODS."""
    idx_path = output_dir() / f"points_{cfg.NUM_POINTS_SUBSAMPLE}_idx.npy"
    subsample_idx = np.load(idx_path) if idx_path.exists() else None
    n_points = cfg.NUM_POINTS_SUBSAMPLE

    for method in cfg.ABLATION_METHODS:
        print(f"\n--- Ablation: {method} ---")
        slerp_aggregate_all(
            n_points=n_points,
            point_indices=subsample_idx,
            method=method,
            force=force,
        )
        agg_path = output_dir() / "aggregated" / f"{method}.npy"
        run_segmentation(agg_path, force=force)
        run_evaluation(method, subsample_idx=subsample_idx)
        run_segmentation_viz(method, force=force)

    run_ablation(subsample_idx=subsample_idx)


def main() -> None:
    set_seed(cfg.RANDOM_SEED)
    ensure = output_dir()
    save_settings_snapshot(ensure, cfg)
    print(f"Output directory: {ensure}")
    print(f"Running steps: {cfg.STEPS}")

    force = cfg.FORCE_RECOMPUTE
    for step in cfg.STEPS:
        run_step(step, force=force)

    # Run ablation if step 9 was included and multiple methods configured
    if 9 in cfg.STEPS and len(cfg.ABLATION_METHODS) > 1:
        print("\nRunning ablation comparison...")
        run_ablation_pipeline(force=force)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
