#!/usr/bin/env python3
"""
Aggregated feature ↔ text similarity heatmap on the subsampled 3D point cloud.

Loads multi-view aggregated DINOtxt vectors (step 7), computes cosine similarity
to a text class prototype (step 8), and saves a jet-colored 3D heatmap PLY + PNG.

Run from repo root:
    conda run -n 3dvision python test/test_aggregated_similarity.py

Edit TARGET_CLASS and AGGREGATION_METHOD below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- config (edit these) ---
TARGET_CLASS = "wall"  # must exist in cfg.TEXT_PROMPTS / text_embeddings/
AGGREGATION_METHOD = "mean"  # file stem under outputs/room2/aggregated/
SHOW_VIEWER = True  # set False on headless/SSH machines
FORCE_RECOMPUTE_TEXT = False  # set True to re-encode text from prompts
# ---------------------------

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_DIR = TEST_DIR / "test_outputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import settings as cfg
from src.data.mesh import (
    gt_labels_path,
    load_info_semantic,
    object_ids_to_class_names,
    output_dir,
)
from src.features.dinotxt_text import (
    build_class_embeddings,
    cosine_similarity_to_text,
    text_embeddings_dir,
)
from src.segmentation.classify import results_dir
from src.utils.io import load_npy, save_npy
from src.viz.gt_labels import save_colored_ply


def run_dir_name() -> str:
    return f"agg_sim_{TARGET_CLASS}_{AGGREGATION_METHOD}"


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
        s=0.4,
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


def resolve_paths() -> tuple[Path, Path, Path]:
    root = output_dir()
    points_path = root / f"points_{cfg.NUM_POINTS_SUBSAMPLE}.npy"
    agg_path = root / "aggregated" / f"{AGGREGATION_METHOD}.npy"
    idx_path = root / f"points_{cfg.NUM_POINTS_SUBSAMPLE}_idx.npy"
    for label, path in (
        ("points", points_path),
        ("aggregated", agg_path),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label} file: {path}\nRun pipeline steps 5 and 7 first."
            )
    return points_path, agg_path, idx_path


def load_text_embedding(class_name: str) -> np.ndarray:
    if class_name not in cfg.TEXT_PROMPTS:
        available = ", ".join(sorted(cfg.TEXT_PROMPTS))
        raise ValueError(f"Unknown class {class_name!r}. Available: {available}")

    emb_path = text_embeddings_dir() / f"{class_name}.npy"
    if emb_path.exists() and not FORCE_RECOMPUTE_TEXT:
        emb = load_npy(emb_path)
        print(f"  loaded text embedding from {emb_path}")
        return emb.astype(np.float32)

    print(f"  encoding '{class_name}' from {len(cfg.TEXT_PROMPTS[class_name])} prompts...")
    embs = build_class_embeddings(class_names=[class_name], force=FORCE_RECOMPUTE_TEXT)
    return embs[class_name]


def load_gt_names(idx_path: Path, n_points: int) -> np.ndarray | None:
    if not idx_path.exists():
        return None
    if not gt_labels_path(cfg.NUM_POINTS_INITIAL).exists():
        return None
    subsample_idx = load_npy(idx_path)
    gt_all = load_npy(gt_labels_path(cfg.NUM_POINTS_INITIAL))
    gt_object_ids = gt_all[subsample_idx]
    obj_to_class, _ = load_info_semantic(cfg.INFO_SEMANTIC_PATH)
    return object_ids_to_class_names(gt_object_ids, obj_to_class)


def gt_diagnostics(
    similarities: np.ndarray,
    gt_names: np.ndarray | None,
    target: str,
) -> dict:
    if gt_names is None:
        return {}
    is_target = gt_names == target
    n_target = int(is_target.sum())
    n_other = int((~is_target).sum())
    out: dict = {
        "gt_target_count": n_target,
        "gt_other_count": n_other,
    }
    if n_target:
        out["similarity_mean_gt_target"] = float(similarities[is_target].mean())
        out["similarity_median_gt_target"] = float(np.median(similarities[is_target]))
    if n_other:
        out["similarity_mean_gt_other"] = float(similarities[~is_target].mean())
    if n_target and n_other:
        out["gt_target_minus_other_mean"] = (
            out["similarity_mean_gt_target"] - out["similarity_mean_gt_other"]
        )
    return out


def prediction_diagnostics(
    similarities: np.ndarray,
    gt_names: np.ndarray | None,
    target: str,
) -> dict:
    pred_path = results_dir() / f"predictions_{AGGREGATION_METHOD}_names.npy"
    if not pred_path.exists() or gt_names is None:
        return {}
    pred_names = np.load(pred_path, allow_pickle=True)
    if len(pred_names) != len(similarities):
        return {}

    pred_target = pred_names == target
    gt_target = gt_names == target
    tp = int((pred_target & gt_target).sum())
    fp = int((pred_target & ~gt_target).sum())
    fn = int((~pred_target & gt_target).sum())
    return {
        "pred_as_target_count": int(pred_target.sum()),
        "tp_target": tp,
        "fp_target": fp,
        "fn_target": fn,
        "similarity_mean_pred_target": float(similarities[pred_target].mean()) if pred_target.any() else None,
        "similarity_mean_pred_not_target": float(similarities[~pred_target].mean())
        if (~pred_target).any()
        else None,
    }


def main() -> None:
    if TARGET_CLASS not in cfg.TEXT_PROMPTS:
        raise ValueError(f"TARGET_CLASS={TARGET_CLASS!r} not in cfg.TEXT_PROMPTS")

    out_dir = OUTPUT_DIR / run_dir_name()
    out_dir.mkdir(parents=True, exist_ok=True)

    points_path, agg_path, idx_path = resolve_paths()
    points = load_npy(points_path).astype(np.float32)
    aggregated = load_npy(agg_path).astype(np.float32)

    if len(points) != len(aggregated):
        raise ValueError(
            f"Point count mismatch: {len(points)} points vs {len(aggregated)} aggregated rows"
        )

    print(f"Loaded {len(points):,} subsampled 3D points from {points_path.name}")
    print(f"Loaded aggregated features {aggregated.shape} from {agg_path.name}")

    text_emb = load_text_embedding(TARGET_CLASS)
    feat_norms = np.linalg.norm(aggregated, axis=1)
    text_norm = float(np.linalg.norm(text_emb))
    print(f"  aggregated L2 norm: mean={feat_norms.mean():.4f}, max={feat_norms.max():.4f}")
    print(f"  text embedding L2 norm: {text_norm:.4f}")

    similarities = cosine_similarity_to_text(aggregated, text_emb)
    sim_min = float(similarities.min())
    sim_max = float(similarities.max())
    sim_mean = float(similarities.mean())
    print(f"\nCosine similarity to '{TARGET_CLASS}' (aggregated, all {len(points):,} points):")
    print(f"  mean = {sim_mean:.4f}, min = {sim_min:.4f}, max = {sim_max:.4f}")
    if sim_max > 1.0 + 1e-3 or sim_min < -1.0 - 1e-3:
        raise RuntimeError(
            "Similarity out of [-1, 1] — vectors may not be L2-normalized. "
            "Re-run aggregation or text encoding."
        )

    gt_names = load_gt_names(idx_path, len(points))
    gt_stats = gt_diagnostics(similarities, gt_names, TARGET_CLASS)
    pred_stats = prediction_diagnostics(similarities, gt_names, TARGET_CLASS)
    if gt_stats:
        print(f"\nGT diagnostics for '{TARGET_CLASS}':")
        print(f"  GT {TARGET_CLASS}: {gt_stats.get('gt_target_count', 0):,} points")
        if "similarity_mean_gt_target" in gt_stats:
            print(f"  mean sim (GT={TARGET_CLASS}): {gt_stats['similarity_mean_gt_target']:.4f}")
        if "similarity_mean_gt_other" in gt_stats:
            print(f"  mean sim (GT≠{TARGET_CLASS}): {gt_stats['similarity_mean_gt_other']:.4f}")
        if "gt_target_minus_other_mean" in gt_stats:
            print(f"  separation (target − other): {gt_stats['gt_target_minus_other_mean']:.4f}")
    if pred_stats:
        print(f"\nPrediction overlap (method={AGGREGATION_METHOD}):")
        print(f"  predicted as {TARGET_CLASS}: {pred_stats.get('pred_as_target_count', 0):,}")
        print(
            f"  TP={pred_stats.get('tp_target', 0):,}, "
            f"FP={pred_stats.get('fp_target', 0):,}, "
            f"FN={pred_stats.get('fn_target', 0):,}"
        )

    sim_vmin = float(np.percentile(similarities, 2))
    sim_vmax = float(np.percentile(similarities, 98))
    colors = similarity_to_jet_colors(similarities, vmin=sim_vmin, vmax=sim_vmax)

    tag = f"{TARGET_CLASS}_{AGGREGATION_METHOD}"
    ply_path = out_dir / f"aggregated_similarity_{tag}.ply"
    png_path = out_dir / f"aggregated_similarity_{tag}.png"
    hist_path = out_dir / f"aggregated_similarity_{tag}_histogram.png"
    sim_npy = out_dir / f"aggregated_similarity_{tag}.npy"

    save_colored_ply(points, colors, ply_path)
    save_scatter_png(
        points,
        colors,
        png_path,
        f"'{TARGET_CLASS}' cosine similarity — aggregated ({AGGREGATION_METHOD}, {len(points):,} pts)",
    )
    save_histogram(
        similarities,
        hist_path,
        f"Cosine similarity to '{TARGET_CLASS}' (aggregated {AGGREGATION_METHOD})",
    )
    save_npy(sim_npy, similarities)

    summary = {
        "target_class": TARGET_CLASS,
        "aggregation_method": AGGREGATION_METHOD,
        "num_points": int(len(points)),
        "similarity_mean": sim_mean,
        "similarity_min": sim_min,
        "similarity_max": sim_max,
        "jet_scale": [sim_vmin, sim_vmax],
        "points_path": str(points_path),
        "aggregated_path": str(agg_path),
        "legend": "jet colormap on subsampled 3D points (multi-view aggregated features)",
        "gt_diagnostics": gt_stats,
        "prediction_diagnostics": pred_stats,
        "outputs": {
            "similarity_npy": str(sim_npy),
            "heatmap_ply": str(ply_path),
            "heatmap_png": str(png_path),
            "histogram_png": str(hist_path),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  {ply_path.name}")
    print(f"  {png_path.name}")
    print(f"  {hist_path.name}")

    if SHOW_VIEWER:
        show_open3d(
            ply_path,
            title=f"'{TARGET_CLASS}' similarity — aggregated ({AGGREGATION_METHOD})",
        )


if __name__ == "__main__":
    main()
