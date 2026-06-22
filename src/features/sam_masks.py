"""
SAM per-frame mask segmentation + per-mask DINOtxt embeddings (step 6b).

For each frame we:
  1. Run SAM automatic mask generation on the RGB image.
  2. Deduplicate near-identical masks (overlap NMS), drop tiny masks.
  3. Compute a DINOtxt embedding per mask = mean of the (already cached, step 6)
     L2-normalized patch tokens inside the mask, re-normalized.
  4. Build a patch-grid seg_map (43x75): each patch -> the smallest mask covering
     it (-1 if none), matching Pe3R's "smallest containing mask wins".

We save only compact artifacts per frame (NOT the full-res masks):
  outputs/{room}/sam/{frame:06d}.npz with
    seg_map  : (PATCH_GRID_H, PATCH_GRID_W) int16   patch -> mask id (-1 = none)
    mask_emb : (M, DINOTXT_FEATURE_DIM)     float32 L2-normalized per-mask embedding
    mask_area: (M,)                          int32   full-res pixel count per mask

Total cache for 2000 frames is a few hundred MB (vs 26 GB for the DINOtxt features),
so the downstream sam_slerp aggregation can hold all of it in RAM at once.

Memory / VRAM notes (tuned for 6 GB VRAM, 24 GB RAM):
  - Only the SAM model lives on the GPU here; DINOtxt features are read from disk
    (one frame at a time, ~13 MB), so VRAM is dedicated to SAM.
  - SAM_USE_AMP runs SAM under fp16 autocast; SAM_POINTS_PER_BATCH bounds the
    mask-decoder batch. On CUDA OOM we retry the frame with a smaller batch.
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
import torch
from src.utils.progress import tqdm

from config import settings as cfg
from src.data.mesh import output_dir
from src.data.replica import load_rgb, load_trajectory
from src.features.dinotxt_features import features_dinotxt_dir, load_frame_features
from src.utils.geometry import normalize_vectors
from src.utils.io import ensure_dir


def sam_masks_dir() -> Path:
    return ensure_dir(output_dir() / "sam")


def sam_frame_path(fi: int) -> Path:
    return sam_masks_dir() / f"{fi:06d}.npz"


class SAMSegmenter:
    """Automatic mask generation with overlap NMS, tuned for low VRAM."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        model_type: str | None = None,
        points_per_side: int | None = None,
        points_per_batch: int | None = None,
        min_mask_area_frac: float | None = None,
        nms_threshold: float | None = None,
        device: str | None = None,
        use_amp: bool | None = None,
    ):
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "segment_anything is required for sam_slerp. Install it with:\n"
                "    pip install segment-anything"
            ) from exc

        self.checkpoint_path = checkpoint_path or cfg.SAM_CHECKPOINT
        self.model_type = model_type or cfg.SAM_MODEL_TYPE
        self.min_mask_area_frac = (
            min_mask_area_frac if min_mask_area_frac is not None else cfg.SAM_MIN_MASK_AREA_FRAC
        )
        self.nms_threshold = nms_threshold if nms_threshold is not None else cfg.SAM_NMS_THRESHOLD
        self.use_amp = use_amp if use_amp is not None else cfg.SAM_USE_AMP

        device = device or cfg.DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            print("  [SAM] CUDA unavailable, falling back to CPU (very slow).")
            device = "cpu"
        self.device = device

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(device)
        sam.eval()
        self._sam = sam

        self.points_per_side = points_per_side or cfg.SAM_POINTS_PER_SIDE
        self.points_per_batch = points_per_batch or cfg.SAM_POINTS_PER_BATCH
        self._build_generator(self.points_per_batch)

    def _build_generator(self, points_per_batch: int) -> None:
        from segment_anything import SamAutomaticMaskGenerator

        self._generator = SamAutomaticMaskGenerator(
            model=self._sam,
            points_per_side=self.points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=cfg.SAM_PRED_IOU_THRESH,
            stability_score_thresh=cfg.SAM_STABILITY_SCORE_THRESH,
            min_mask_region_area=cfg.SAM_MIN_MASK_REGION_AREA,
        )

    def _mask_nms(self, masks: list[np.ndarray]) -> list[np.ndarray]:
        """Remove near-duplicate masks via pairwise min-overlap suppression."""
        if not masks:
            return masks
        keep: list[int] = []
        suppressed: set[int] = set()
        for i in range(len(masks)):
            if i in suppressed:
                continue
            keep.append(i)
            ai = masks[i].sum()
            for j in range(i + 1, len(masks)):
                if j in suppressed:
                    continue
                intersection = np.logical_and(masks[i], masks[j]).sum()
                min_overlap = min(
                    intersection / max(ai, 1),
                    intersection / max(masks[j].sum(), 1),
                )
                if min_overlap > self.nms_threshold:
                    suppressed.add(j)
        return [masks[i] for i in keep]

    @torch.inference_mode()
    def _generate(self, img: np.ndarray) -> list[dict]:
        """Run AMG with fp16 autocast and a CUDA-OOM retry that shrinks the batch."""
        ppb = self.points_per_batch
        while True:
            try:
                if self.use_amp and self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        return self._generator.generate(img)
                return self._generator.generate(img)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if ppb <= 8:
                    raise
                ppb = max(8, ppb // 2)
                print(f"  [SAM] CUDA OOM -> retrying with points_per_batch={ppb}")
                self._build_generator(ppb)

    def segment_image(self, img: np.ndarray) -> list[np.ndarray]:
        """Boolean masks (H, W) for one RGB image, sorted by area descending."""
        img_area = img.shape[0] * img.shape[1]
        sam_output = self._generate(img)
        masks = [
            entry["segmentation"].astype(bool)
            for entry in sam_output
            if entry["segmentation"].sum() / img_area >= self.min_mask_area_frac
        ]
        masks.sort(key=lambda m: m.sum(), reverse=True)
        masks = self._mask_nms(masks)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return masks


def _mask_to_patch_grid(mask: np.ndarray) -> np.ndarray:
    """Downsample a full-res boolean mask to the DINOtxt patch grid (H_p, W_p).

    The mask is padded to PAD_HEIGHT first so patch rows align exactly with the
    DINOtxt feature grid (which is computed on the bottom-padded image).
    """
    h, w = mask.shape
    if h < cfg.PAD_HEIGHT:
        mask = np.pad(mask, ((0, cfg.PAD_HEIGHT - h), (0, 0)), constant_values=False)
    grid = cv2.resize(
        mask.astype(np.uint8),
        (cfg.PATCH_GRID_W, cfg.PATCH_GRID_H),
        interpolation=cv2.INTER_NEAREST,
    )
    return grid.astype(bool)


def _slerp_pair_overlap(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Minimal SLERP for intra-frame overlap adjustment (reference parity)."""
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return (1.0 - t) * v0 + t * v1
    u0, u1 = v0 / n0, v1 / n1
    dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    if abs(dot) > 0.9995:
        out = (1.0 - t) * u0 + t * u1
        return out / (np.linalg.norm(out) + 1e-12)
    omega = np.arccos(dot)
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * u0 + np.sin(t * omega) / so * u1


def _overlap_slerp(
    masks: list[np.ndarray],
    embeddings: list[np.ndarray],
    threshold: float = 0.025,
) -> list[np.ndarray]:
    """Pe3R-style: pull a smaller mask's embedding toward an overlapping larger one."""
    adjusted = [e.copy() for e in embeddings]
    areas = [float(m.sum()) for m in masks]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if areas[i] == 0 or areas[j] == 0:
                continue
            inter = float(np.logical_and(masks[i], masks[j]).sum())
            if min(inter / areas[i], inter / areas[j]) > threshold:
                t = areas[i] / (areas[i] + areas[j])
                adjusted[j] = _slerp_pair_overlap(adjusted[j], adjusted[i], t)
    return adjusted


def _mask_embeddings(
    masks: list[np.ndarray],
    feat_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-mask mean of patch tokens + patch-grid seg_map (smallest mask wins).

    Returns (mask_emb (M, D) L2-normalized, mask_area (M,), seg_map (H_p, W_p) int16).
    """
    dim = feat_flat.shape[1]
    seg_map = np.full((cfg.PATCH_GRID_H, cfg.PATCH_GRID_W), -1, dtype=np.int16)
    embs = np.zeros((len(masks), dim), dtype=np.float32)
    areas = np.zeros(len(masks), dtype=np.int32)

    # masks are sorted largest-first; assign in that order so smaller masks
    # (later indices) overwrite -> seg_map holds the smallest covering mask.
    for idx, mask in enumerate(masks):
        areas[idx] = int(mask.sum())
        grid = _mask_to_patch_grid(mask)
        patch_ids = np.flatnonzero(grid)
        if patch_ids.size > 0:
            embs[idx] = feat_flat[patch_ids].mean(axis=0)
            seg_map.reshape(-1)[patch_ids] = idx

    embs = normalize_vectors(embs.astype(np.float64)).astype(np.float32)
    return embs, areas, seg_map


def build_sam_masks(force: bool = False, device: str | None = None) -> Path:
    """Step 6b: SAM masks + per-mask DINOtxt embeddings for all frames (cached)."""
    out_dir = sam_masks_dir()
    feat_dir = features_dinotxt_dir()
    poses = load_trajectory()
    n_frames = len(poses)

    if not force:
        if all(sam_frame_path(fi).exists() for fi in range(n_frames)):
            print(f"All {n_frames} SAM mask files exist, skipping.")
            return out_dir

    overlap = cfg.SAM_OVERLAP_SLERP
    print(
        f"Building SAM masks for {n_frames} frames "
        f"(model={cfg.SAM_MODEL_TYPE}, points_per_side={cfg.SAM_POINTS_PER_SIDE}, "
        f"amp={cfg.SAM_USE_AMP}, overlap_slerp={overlap})"
    )
    segmenter = SAMSegmenter(device=device)

    total_masks = 0
    for fi in tqdm(range(n_frames), desc="SAM masks"):
        out_path = sam_frame_path(fi)
        if out_path.exists() and not force:
            continue

        feat_path = feat_dir / f"{fi:06d}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(
                f"DINOtxt features missing for frame {fi}: {feat_path}. Run step 6 first."
            )

        img = load_rgb(fi)
        masks = segmenter.segment_image(img)
        feat = load_frame_features(fi, feat_dir)  # (H_p, W_p, D)
        feat_flat = feat.reshape(-1, feat.shape[-1]).astype(np.float32)

        if len(masks) == 0:
            seg_map = np.full((cfg.PATCH_GRID_H, cfg.PATCH_GRID_W), -1, dtype=np.int16)
            mask_emb = np.zeros((0, feat_flat.shape[1]), dtype=np.float32)
            mask_area = np.zeros(0, dtype=np.int32)
        else:
            mask_emb, mask_area, seg_map = _mask_embeddings(masks, feat_flat)
            if overlap:
                adj = _overlap_slerp(masks, [mask_emb[i] for i in range(len(masks))])
                mask_emb = normalize_vectors(np.stack(adj).astype(np.float64)).astype(np.float32)

        np.savez_compressed(
            out_path, seg_map=seg_map, mask_emb=mask_emb, mask_area=mask_area
        )
        total_masks += len(masks)
        del img, masks, feat, feat_flat

    print(f"Saved SAM masks to {out_dir} ({total_masks} masks across {n_frames} frames)")
    return out_dir


def load_sam_frame(fi: int) -> dict[str, np.ndarray]:
    """Load compact SAM artifacts for one frame."""
    with np.load(sam_frame_path(fi)) as data:
        return {
            "seg_map": data["seg_map"],
            "mask_emb": data["mask_emb"],
            "mask_area": data["mask_area"],
        }


def main():
    # Smoke test of the compact-artifact math without SAM/GPU: fake 3 masks on
    # the patch grid and confirm seg_map / embedding shapes.
    rng = np.random.default_rng(0)
    h, w = cfg.PAD_HEIGHT, cfg.IMAGE_WIDTH
    masks = []
    big = np.zeros((h, w), dtype=bool)
    big[:, : w // 2] = True
    small = np.zeros((h, w), dtype=bool)
    small[: h // 3, : w // 4] = True
    masks = [big, small]  # already largest-first
    feat_flat = rng.standard_normal((cfg.PATCH_GRID_H * cfg.PATCH_GRID_W, cfg.DINOTXT_FEATURE_DIM)).astype(np.float32)
    emb, area, seg = _mask_embeddings(masks, feat_flat)
    print(f"mask_emb {emb.shape}, norms={np.linalg.norm(emb, axis=1)}")
    print(f"areas={area}, seg_map unique={np.unique(seg)}")


if __name__ == "__main__":
    main()
