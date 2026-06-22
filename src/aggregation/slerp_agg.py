"""Spherical linear interpolation and Fréchet mean aggregation."""

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
from src.utils.progress import tqdm

from config import settings as cfg
from src.aggregation.mean_agg import (
    aggregated_dir,
    aggregate_mean_vectorized,
    compute_mean_observation_distance,
    iter_point_features_chunked,
    mean_aggregate,
)
from src.data.projection import load_visibility
from src.data.replica import load_camera_params
from src.utils.geometry import normalize_vectors
from src.utils.io import save_npy


def slerp(v0: np.ndarray, v1: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors."""
    v0 = normalize_vectors(v0.reshape(1, -1)).squeeze(0)
    v1 = normalize_vectors(v1.reshape(1, -1)).squeeze(0)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    if dot > 0.9995:
        out = (1 - t) * v0 + t * v1
        return normalize_vectors(out.reshape(1, -1)).squeeze(0)
    if dot < -0.9995:
        out = (1 - t) * v0 + t * (-v1)
        return normalize_vectors(out.reshape(1, -1)).squeeze(0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    s0 = np.sin((1 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta
    out = s0 * v0 + s1 * v1
    return normalize_vectors(out.reshape(1, -1)).squeeze(0)


def iterative_slerp(feats: np.ndarray) -> np.ndarray:
    """Binary-tree iterative slerp over normalized feature vectors."""
    if len(feats) == 0:
        return np.zeros(cfg.DINOTXT_FEATURE_DIM, dtype=np.float32)
    if len(feats) == 1:
        return normalize_vectors(feats[0].reshape(1, -1)).squeeze(0).astype(np.float32)

    vecs = [normalize_vectors(f.reshape(1, -1)).squeeze(0) for f in feats]
    while len(vecs) > 1:
        next_vecs = []
        for i in range(0, len(vecs), 2):
            if i + 1 < len(vecs):
                next_vecs.append(slerp(vecs[i], vecs[i + 1], 0.5))
            else:
                next_vecs.append(vecs[i])
        vecs = next_vecs
    return vecs[0].astype(np.float32)


def frechet_mean(
    feats: np.ndarray,
    max_iter: int | None = None,
    tol: float | None = None,
) -> np.ndarray:
    """Fréchet mean on the unit hypersphere."""
    max_iter = max_iter if max_iter is not None else cfg.FRECHET_MAX_ITER
    tol = tol if tol is not None else cfg.FRECHET_TOL

    if len(feats) == 0:
        return np.zeros(cfg.DINOTXT_FEATURE_DIM, dtype=np.float32)
    if len(feats) == 1:
        return normalize_vectors(feats[0].reshape(1, -1)).squeeze(0).astype(np.float32)

    normed = normalize_vectors(feats)
    mu = normalize_vectors(normed.mean(axis=0).reshape(1, -1)).squeeze(0)

    for _ in range(max_iter):
        dots = np.clip(normed @ mu, -1.0, 1.0)
        thetas = np.arccos(dots)
        if np.all(thetas < 1e-8):
            break
        # tangent vectors on sphere
        coeffs = np.where(thetas > 1e-8, thetas / np.sin(thetas), 0.0)
        tangent = (coeffs[:, None] * (normed - (dots[:, None] * mu))).sum(axis=0)
        norm_t = np.linalg.norm(tangent)
        if norm_t < tol:
            break
        mu = normalize_vectors((mu + tangent).reshape(1, -1)).squeeze(0)

    return mu.astype(np.float32)


def _slerp_pair(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two vectors (reference parity).

    Both inputs are normalized internally. Near-parallel *and* near-antiparallel
    vectors (``|dot| > 0.9995``) fall back to normalized linear interpolation
    without flipping either vector; degenerate (near-zero) inputs use plain lerp.
    """
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return (1.0 - t) * v0 + t * v1

    u0 = v0 / n0
    u1 = v1 / n1

    dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))

    if abs(dot) > 0.9995:
        result = (1.0 - t) * u0 + t * u1
        return result / (np.linalg.norm(result) + 1e-12)

    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    coeff0 = np.sin((1.0 - t) * omega) / sin_omega
    coeff1 = np.sin(t * omega) / sin_omega
    return coeff0 * u0 + coeff1 * u1


def _weight_distance_only(distance: float, mean_distance: float) -> float:
    d_norm = distance / max(mean_distance, 1e-12)
    return 1.0 / (1.0 + d_norm)


def _weight_angle_only(angle: float) -> float:
    return max(0.0, float(np.cos(angle)))


def _weight_distance_and_angle(distance: float, angle: float, mean_distance: float) -> float:
    return _weight_distance_only(distance, mean_distance) * _weight_angle_only(angle)


def _slerp_from_weights(feats: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Sequential weighted SLERP over feature vectors (reference parity)."""
    if len(feats) == 0:
        return np.zeros(cfg.DINOTXT_FEATURE_DIM, dtype=np.float32)
    if len(feats) == 1:
        return normalize_vectors(feats[0].reshape(1, -1)).squeeze(0).astype(np.float32)

    w = weights.astype(np.float64)
    total = w.sum()
    if total < 1e-12:
        w = np.full(len(w), 1.0 / len(w))
    else:
        w = w / total

    # Ascending weight order; stable to match the reference's sorted() tie-breaking.
    order = np.argsort(w, kind="stable")
    result = feats[order[0]].astype(np.float64)
    acc_w = w[order[0]]

    for idx in order[1:]:
        t = w[idx] / (acc_w + w[idx])
        result = _slerp_pair(result, feats[idx].astype(np.float64), float(t))
        acc_w += w[idx]

    return result.astype(np.float32)


def weighted_slerp_aggregate(
    feats: np.ndarray,
    distances: np.ndarray,
    angles: np.ndarray,
    mean_distance: float,
) -> np.ndarray:
    """SLERP weighted by normalised distance and viewing angle (closer, head-on = higher)."""
    if len(feats) == 0:
        return np.zeros(cfg.DINOTXT_FEATURE_DIM, dtype=np.float32)
    weights = np.array(
        [_weight_distance_and_angle(float(d), float(a), mean_distance) for d, a in zip(distances, angles)],
        dtype=np.float64,
    )
    return _slerp_from_weights(feats, weights)


def aggregate_fn(method: str):
    if method == "mean":
        return mean_aggregate
    if method == "slerp":
        return iterative_slerp
    if method == "frechet":
        return frechet_mean
    if method == "weighted_slerp":
        return weighted_slerp_aggregate
    raise ValueError(f"Unknown aggregation method: {method}")


def _load_sam_cache(frames: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """Load compact SAM artifacts for the given frames into RAM (~0.1-0.3 GB total)."""
    from src.features.sam_masks import load_sam_frame, sam_frame_path

    cache: dict[int, dict[str, np.ndarray]] = {}
    for fi in tqdm(np.unique(frames), desc="Load SAM cache"):
        fi = int(fi)
        if sam_frame_path(fi).exists():
            cache[fi] = load_sam_frame(fi)
    return cache


def sam_slerp_aggregate_all(
    n_points: int,
    point_indices: np.ndarray | None,
    vis: dict[str, np.ndarray],
) -> np.ndarray:
    """
    SAM mask-level area-weighted SLERP aggregation.

    Each observation contributes the DINOtxt embedding of the SAM mask that
    covers its patch (looked up via the cached per-frame seg_map); the per-point
    result is an area-weighted SLERP over those mask embeddings. The compact SAM
    cache fits in RAM, so this is fast and memory-light (no 26 GB feature re-read).
    """
    dim = cfg.DINOTXT_FEATURE_DIM
    offsets = vis["point_obs_offsets"]
    obs_frame = vis["obs_frame_idx"]
    obs_px = vis["obs_patch_x"]
    obs_py = vis["obs_patch_y"]

    sam_cache = _load_sam_cache(obs_frame)

    aggregated = np.zeros((n_points, dim), dtype=np.float32)
    for slot in tqdm(range(n_points), desc="Aggregate (sam_slerp)"):
        orig = int(point_indices[slot]) if point_indices is not None else slot
        start = int(offsets[orig])
        end = int(offsets[orig + 1])
        if start == end:
            continue

        feats: list[np.ndarray] = []
        weights: list[float] = []
        for k in range(start, end):
            fi = int(obs_frame[k])
            entry = sam_cache.get(fi)
            if entry is None:
                continue
            mask_id = int(entry["seg_map"][int(obs_py[k]), int(obs_px[k])])
            if mask_id < 0:
                continue
            feats.append(entry["mask_emb"][mask_id])
            weights.append(float(entry["mask_area"][mask_id]))

        if not feats:
            continue
        aggregated[slot] = _slerp_from_weights(
            np.stack(feats, axis=0), np.asarray(weights, dtype=np.float64)
        )
    return aggregated


def aggregate_all_points(
    n_points: int,
    point_indices: np.ndarray | None = None,
    vis: dict | None = None,
    method: str | None = None,
    force: bool = False,
) -> Path:
    """Step 7: aggregate using slerp or frechet."""
    method = method or cfg.AGGREGATION_METHOD
    out_path = aggregated_dir() / f"{method}.npy"
    if out_path.exists() and not force:
        print(f"Aggregated features exist at {out_path}")
        return out_path

    vis = vis or load_visibility()

    if method == "mean":
        aggregated = aggregate_mean_vectorized(vis, n_points, point_indices)
    elif method == "sam_slerp":
        from src.features.sam_masks import build_sam_masks, sam_frame_path

        n_frames = int(vis["num_frames"][0])
        if not all(sam_frame_path(fi).exists() for fi in range(n_frames)):
            print("  SAM mask cache incomplete — building it first (one-time, GPU).")
            build_sam_masks(force=False)
        aggregated = sam_slerp_aggregate_all(n_points, point_indices, vis)
    elif method == "weighted_slerp":
        cam = load_camera_params()
        mean_distance = compute_mean_observation_distance(vis, cam)
        chunk_obs = cfg.AGGREGATION_MAX_OBS_PER_CHUNK
        est_pts = max(1, chunk_obs // 415)  # ~415 obs/point on production subsample
        est_chunks = max(1, (n_points + est_pts - 1) // est_pts)
        print(f"  weighted_slerp mean observation distance: {mean_distance:.4f} m")
        print(
            f"  chunk size: {chunk_obs:,} obs (~{est_pts:,} points/chunk, "
            f"~{est_chunks} disk passes over {cfg.NUM_FRAMES or 2000} frames)"
        )
        aggregated = np.zeros((n_points, cfg.DINOTXT_FEATURE_DIM), dtype=np.float32)
        for slot, feats, distances, angles in tqdm(
            iter_point_features_chunked(
                vis, n_points, point_indices, cam=cam, max_obs_per_chunk=chunk_obs
            ),
            desc=f"Aggregate ({method})",
            total=n_points,
        ):
            aggregated[slot] = weighted_slerp_aggregate(feats, distances, angles, mean_distance)
    else:
        fn = aggregate_fn(method)
        aggregated = np.zeros((n_points, cfg.DINOTXT_FEATURE_DIM), dtype=np.float32)
        for slot, feats, _d, _a in tqdm(
            iter_point_features_chunked(
                vis, n_points, point_indices, cam=None, max_obs_per_chunk=cfg.AGGREGATION_MAX_OBS_PER_CHUNK
            ),
            desc=f"Aggregate ({method})",
            total=n_points,
        ):
            aggregated[slot] = fn(feats)

    save_npy(out_path, aggregated)
    print(f"Saved aggregated features to {out_path}")
    return out_path


def main():
    rng = np.random.default_rng(0)
    feats = normalize_vectors(rng.standard_normal((5, cfg.DINOTXT_FEATURE_DIM)))
    distances = np.array([1.0, 2.0, 1.5, 3.0, 0.8], dtype=np.float32)
    angles = np.array([0.1, 0.8, 0.3, 1.2, 0.05], dtype=np.float32)
    m_slerp = iterative_slerp(feats)
    m_frechet = frechet_mean(feats)
    m_weighted = weighted_slerp_aggregate(feats, distances, angles, float(distances.mean()))
    print(
        f"slerp norm={np.linalg.norm(m_slerp):.4f}, "
        f"frechet norm={np.linalg.norm(m_frechet):.4f}, "
        f"weighted_slerp norm={np.linalg.norm(m_weighted):.4f}"
    )


if __name__ == "__main__":
    main()
