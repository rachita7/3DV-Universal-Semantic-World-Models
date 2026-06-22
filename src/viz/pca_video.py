"""PCA-comparison video: dense per-frame features vs. multi-view aggregation.

Layout adapts to the number of aggregation methods:
  1 method  -> [Original | Dense PCA | method]
  2 methods -> 2x2 [Original | Dense PCA] / [method 1 | method 2]

The dense panel PCA-colors per-frame patch features (DINOtxt or ViT-S/16); each
method panel projects aggregated per-point embeddings back into the frame via
visibility.npz and overlays them on the dimmed original. Two methods share one
PCA basis so colours stay comparable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import cv2
import numpy as np
from src.utils.progress import tqdm

from config import settings as cfg
from src.aggregation.mean_agg import aggregated_dir, build_subsample_slot_map
from src.data.mesh import output_dir
from src.data.replica import load_camera_params, load_rgb, load_trajectory
from src.data.projection import load_visibility
from src.features.dino_backbone import features_vits16_dir
from src.features.dinotxt_features import features_dinotxt_dir
from src.utils.geometry import normalize_vectors
from src.utils.io import ensure_dir, load_npy


def pca_video_dir() -> Path:
    return ensure_dir(output_dir() / "pca_video")


def _fit_pca(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 3-component PCA via SVD. Returns (mean, components[3, D])."""
    x = features.astype(np.float64)
    mean = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt[:3]


def _project(features: np.ndarray, mean: np.ndarray, comp: np.ndarray) -> np.ndarray:
    return (features.astype(np.float64) - mean) @ comp.T


def _percentile_bounds(
    proj: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.percentile(proj, lo_pct, axis=0),
        np.percentile(proj, hi_pct, axis=0),
    )


def _project_to_rgb(
    features: np.ndarray, mean: np.ndarray, comp: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    proj = _project(features, mean, comp)
    rgb = np.zeros_like(proj)
    for c in range(3):
        rgb[:, c] = (proj[:, c] - lo[c]) / (hi[c] - lo[c] + 1e-8)
    return np.clip(rgb, 0.0, 1.0)


def _dense_feature_dir(source: str) -> tuple[Path, int]:
    if source == "vits16":
        return features_vits16_dir(), cfg.VITS16_FEATURE_DIM
    if source == "dinotxt":
        return features_dinotxt_dir(), cfg.DINOTXT_FEATURE_DIM
    raise ValueError(f"Unknown dense source: {source!r} (use 'dinotxt' or 'vits16')")


def _frame_feature_path(feature_dir: Path, fi: int) -> Path:
    return feature_dir / f"{fi:06d}.npy"


def _collect_dense_samples(
    feature_dir: Path, frames: list[int], dim: int, samples_per_frame: int = 500
) -> np.ndarray:
    rng = np.random.default_rng(42)
    parts: list[np.ndarray] = []
    for fi in frames:
        path = _frame_feature_path(feature_dir, fi)
        if not path.exists():
            continue
        feat = np.load(path).reshape(-1, dim)
        k = min(samples_per_frame, len(feat))
        idx = rng.choice(len(feat), k, replace=False)
        parts.append(feat[idx])
    if not parts:
        raise FileNotFoundError(
            f"No dense feature files found in {feature_dir}. Run the feature-extraction "
            f"step first (step 6 for dinotxt, step 3 for vits16)."
        )
    return np.vstack(parts)


def _make_dense_frame(
    feature_dir: Path,
    fi: int,
    dim: int,
    mean: np.ndarray,
    comp: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> np.ndarray | None:
    path = _frame_feature_path(feature_dir, fi)
    if not path.exists():
        return None
    feat = np.load(path)  # (H_p, W_p, dim)
    h_p, w_p, _ = feat.shape
    flat = normalize_vectors(feat.reshape(-1, dim).astype(np.float64))
    rgb = _project_to_rgb(flat, mean, comp, lo, hi).reshape(h_p, w_p, 3)
    img = (rgb * 255).astype(np.uint8)
    return cv2.resize(img, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)


def _build_frame_to_obs(
    vis: dict[str, np.ndarray], slot_map: np.ndarray, valid_slot: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Invert CSR visibility into frame -> (patch_x, patch_y, slot).

    Keeps only observations whose point maps to a slot with a non-zero embedding.
    """
    slots_all = slot_map[vis["obs_point_idx"].astype(np.int64)]
    keep = slots_all >= 0
    slots = slots_all[keep]
    keep2 = valid_slot[slots]

    slots = slots[keep2].astype(np.int64)
    frames = vis["obs_frame_idx"][keep][keep2].astype(np.int64)
    px = vis["obs_patch_x"][keep][keep2].astype(np.int64)
    py = vis["obs_patch_y"][keep][keep2].astype(np.int64)

    if len(frames) == 0:
        return {}

    order = np.argsort(frames, kind="stable")
    frames, px, py, slots = frames[order], px[order], py[order], slots[order]
    uniq, starts = np.unique(frames, return_index=True)
    ends = np.append(starts[1:], len(frames))
    return {
        int(f): (px[s:e], py[s:e], slots[s:e])
        for f, s, e in zip(uniq, starts, ends)
    }


def _compose_overlay(
    valid_patch_flat: np.ndarray,
    valid_rgb: np.ndarray,
    orig_resized: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Overlay colored patches on a dimmed original, dilated for visibility."""
    n_patches = cfg.PATCH_GRID_H * cfg.PATCH_GRID_W
    pca_img = np.zeros((n_patches, 3), dtype=np.float32)
    has_data = np.zeros(n_patches, dtype=bool)
    pca_img[valid_patch_flat] = valid_rgb.astype(np.float32)
    has_data[valid_patch_flat] = True

    pca_img = pca_img.reshape(cfg.PATCH_GRID_H, cfg.PATCH_GRID_W, 3)
    mask = has_data.reshape(cfg.PATCH_GRID_H, cfg.PATCH_GRID_W).astype(np.float32)

    pca_up = cv2.resize(pca_img, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
    mask_up = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

    kernel = np.ones((3, 3), np.uint8)
    mask_up = cv2.dilate((mask_up > 0).astype(np.uint8), kernel, iterations=1).astype(np.float32)

    dimmed = orig_resized.astype(np.float32) / 255.0 * 0.3
    alpha = np.stack([mask_up] * 3, axis=-1)
    blended = pca_up * alpha + dimmed * (1.0 - alpha)
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def _make_sparse_frame(
    agg: np.ndarray,
    obs: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    mean: np.ndarray,
    comp: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    orig_resized: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> tuple[np.ndarray, int]:
    dimmed = (orig_resized.astype(np.float32) * 0.3).astype(np.uint8)
    if obs is None:
        return dimmed, 0

    px, py, slots = obs
    patch_flat = py * cfg.PATCH_GRID_W + px
    embs = agg[slots]

    # Multiple points can land on one patch; average their embeddings.
    uniq_patch, inv = np.unique(patch_flat, return_inverse=True)
    sums = np.zeros((len(uniq_patch), agg.shape[1]), dtype=np.float64)
    counts = np.zeros(len(uniq_patch), dtype=np.float64)
    np.add.at(sums, inv, embs.astype(np.float64))
    np.add.at(counts, inv, 1.0)
    mean_embs = sums / counts[:, None]

    valid_rgb = _project_to_rgb(mean_embs, mean, comp, lo, hi)
    frame = _compose_overlay(uniq_patch, valid_rgb, orig_resized, frame_h, frame_w)
    return frame, len(slots)


def generate_pca_video(
    methods: list[str],
    *,
    dense_source: str | None = None,
    output_path: str | Path | None = None,
    fps: int | None = None,
    frame_h: int | None = None,
    max_frames: int | None = None,
    frame_stride: int = 1,
    fit_frames: int | None = None,
) -> Path:
    """Render a PCA comparison video over the trajectory frames.

    methods is 1 or 2 aggregation names, each needing a saved
    outputs/<room>/aggregated/<method>.npy; None args fall back to PCA_VIDEO_*
    settings. dense_source picks the dense panel features ("dinotxt" | "vits16").
    """
    n_methods = len(methods)
    if n_methods not in (1, 2):
        raise ValueError(f"Expected 1 or 2 methods, got {n_methods}")

    dense_source = dense_source or cfg.PCA_VIDEO_DENSE_SOURCE
    fps = fps if fps is not None else cfg.PCA_VIDEO_FPS
    frame_h = frame_h if frame_h is not None else cfg.PCA_VIDEO_FRAME_H
    max_frames = max_frames if max_frames is not None else cfg.PCA_VIDEO_MAX_FRAMES
    fit_frames = fit_frames if fit_frames is not None else cfg.PCA_VIDEO_FIT_FRAMES
    if output_path is None:
        output_path = pca_video_dir() / f"pca_comparison_{'_vs_'.join(methods)}.mp4"
    output_path = Path(output_path)

    # Load aggregated embeddings, visibility, and the point -> slot mapping.
    aggs: list[np.ndarray] = []
    for m in methods:
        agg_path = aggregated_dir() / f"{m}.npy"
        if not agg_path.exists():
            raise FileNotFoundError(
                f"Aggregated features not found: {agg_path}. Run step 7 for method '{m}' first."
            )
        aggs.append(load_npy(agg_path).astype(np.float32))

    vis = load_visibility()
    n_orig = int(vis["num_points"][0])
    n_pts = aggs[0].shape[0]
    if any(a.shape[0] != n_pts for a in aggs):
        raise ValueError("All methods must have the same number of aggregated points.")
    dim_agg = aggs[0].shape[1]

    idx_path = output_dir() / f"points_{cfg.NUM_POINTS_SUBSAMPLE}_idx.npy"
    if idx_path.exists() and load_npy(idx_path).shape[0] == n_pts:
        point_indices = load_npy(idx_path).astype(np.int64)
    else:
        point_indices = None
    slot_map = build_subsample_slot_map(n_orig, point_indices, n_pts)

    # Render only frames that have dense features on disk.
    poses = load_trajectory()
    n_frames = len(poses)
    feature_dir, dense_dim = _dense_feature_dir(dense_source)
    frames = [
        fi for fi in range(0, n_frames, max(1, frame_stride))
        if _frame_feature_path(feature_dir, fi).exists()
    ]
    if not frames:
        raise FileNotFoundError(
            f"No dense feature files in {feature_dir}. Run the relevant extraction step first."
        )
    if max_frames is not None:
        frames = frames[:max_frames]

    cam = load_camera_params()
    aspect = cam.w / cam.h
    frame_w = int(frame_h * aspect)
    frame_w += frame_w % 2
    frame_h += frame_h % 2

    print(
        f"[pca_video] {len(frames)} frames, methods={methods}, "
        f"dense_source={dense_source} ({dense_dim}d), agg_dim={dim_agg}"
    )

    # Dense PCA basis: fit on an evenly-spaced subset of frames.
    print("[pca_video] Fitting dense PCA ...")
    n_fit = min(fit_frames, len(frames))
    fit_idx = np.linspace(0, len(frames) - 1, n_fit).astype(int)
    fit_frame_ids = [frames[i] for i in np.unique(fit_idx)]
    dense_samples = normalize_vectors(
        _collect_dense_samples(feature_dir, fit_frame_ids, dense_dim).astype(np.float64)
    )
    dense_mean, dense_comp = _fit_pca(dense_samples)
    dense_lo, dense_hi = _percentile_bounds(_project(dense_samples, dense_mean, dense_comp))

    # Aggregated PCA basis: shared across methods so colours match.
    print("[pca_video] Fitting aggregated PCA ...")
    valid_slots = [np.linalg.norm(a, axis=1) > 1e-8 for a in aggs]
    fit_parts = [normalize_vectors(a[v].astype(np.float64)) for a, v in zip(aggs, valid_slots)]
    fit_parts = [p for p in fit_parts if len(p) > 0]
    if fit_parts:
        agg_fit = np.vstack(fit_parts)
        agg_mean, agg_comp = _fit_pca(agg_fit)
        agg_lo, agg_hi = _percentile_bounds(_project(agg_fit, agg_mean, agg_comp))
    else:
        agg_mean, agg_comp, agg_lo, agg_hi = dense_mean, dense_comp, dense_lo, dense_hi

    frame_to_obs = [_build_frame_to_obs(vis, slot_map, v) for v in valid_slots]

    if n_methods == 1:
        vid_w, vid_h = frame_w * 3, frame_h
    else:
        vid_w, vid_h = frame_w * 2, frame_h * 2

    ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h)
    )
    font = cv2.FONT_HERSHEY_SIMPLEX

    for fi in tqdm(frames, desc="[pca_video] frames"):
        orig = load_rgb(fi)
        orig_resized = cv2.resize(orig, (frame_w, frame_h))
        dimmed = (orig_resized.astype(np.float32) * 0.3).astype(np.uint8)

        dense = _make_dense_frame(
            feature_dir, fi, dense_dim, dense_mean, dense_comp,
            dense_lo, dense_hi, frame_h, frame_w,
        )
        if dense is None:
            dense = dimmed.copy()

        method_frames: list[np.ndarray] = []
        pts_counts: list[int] = []
        for mi in range(n_methods):
            obs = frame_to_obs[mi].get(fi)
            sparse, n_pts_frame = _make_sparse_frame(
                aggs[mi], obs, agg_mean, agg_comp, agg_lo, agg_hi,
                orig_resized, frame_h, frame_w,
            )
            method_frames.append(sparse)
            pts_counts.append(n_pts_frame)

        if n_methods == 1:
            frame = np.hstack([orig_resized, dense, method_frames[0]])
            cv2.putText(frame, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Dense {dense_source} PCA", (frame_w + 10, 30), font, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, methods[0], (frame_w * 2 + 10, 30), font, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"frame{fi:06d}  ({pts_counts[0]} pts)", (10, frame_h - 12), font, 0.45, (180, 180, 180), 1)
        else:
            top = np.hstack([orig_resized, dense])
            bot = np.hstack([method_frames[0], method_frames[1]])
            frame = np.vstack([top, bot])
            cv2.putText(frame, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Dense {dense_source} PCA", (frame_w + 10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, methods[0], (10, frame_h + 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, methods[1], (frame_w + 10, frame_h + 30), font, 0.7, (255, 255, 255), 2)
            info = f"frame{fi:06d}  ({methods[0]}:{pts_counts[0]}  {methods[1]}:{pts_counts[1]})"
            cv2.putText(frame, info, (10, vid_h - 12), font, 0.45, (180, 180, 180), 1)

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"[pca_video] Saved -> {output_path}")
    return output_path


def run_pca_video(methods: list[str] | str | None = None, force: bool = False) -> Path:
    """Pipeline entry point (step 10): build the PCA comparison video."""
    methods = methods if methods is not None else cfg.PCA_VIDEO_METHODS
    methods = methods or [cfg.AGGREGATION_METHOD]
    if isinstance(methods, str):
        methods = [methods]
    methods = list(methods)[:2]

    out_path = pca_video_dir() / f"pca_comparison_{'_vs_'.join(methods)}.mp4"
    if out_path.exists() and not force:
        print(f"PCA video exists at {out_path}")
        return out_path
    return generate_pca_video(methods, output_path=out_path)


def main() -> None:
    run_pca_video(force=True)


if __name__ == "__main__":
    main()
