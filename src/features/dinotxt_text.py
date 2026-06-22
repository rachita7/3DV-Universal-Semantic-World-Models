"""DINOtxt text encoder: prompt ensembles to class prototype vectors."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import numpy as np
import torch

from config import settings as cfg
from src.data.mesh import load_info_semantic, output_dir
from src.utils.geometry import normalize_vectors
from src.utils.io import ensure_dir, save_npy

_DINOV3_ROOT = Path(cfg.DINOV3_REPO).resolve()
if str(_DINOV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_DINOV3_ROOT))


def text_embeddings_dir() -> Path:
    return ensure_dir(output_dir() / "text_embeddings")


def load_dinotxt(device: str | None = None):
    from src.features.model_loaders import load_dinotxt as _load

    return _load(device)


@torch.inference_mode()
def encode_prompts(model, tokenizer, prompts: list[str], device: str) -> np.ndarray:
    """Encode prompts; return L2-normalized patch-aligned 1024-d embeddings (P, D)."""
    tokens = tokenizer.tokenize(prompts).to(device)
    text_full = model.encode_text(tokens, normalize=False)
    # Second half of the 2048-d joint embedding aligns with image_patch_tokens.
    emb = text_full[:, cfg.DINOTXT_FEATURE_DIM :]
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.cpu().numpy().astype(np.float32)


def ensemble_class_embedding(model, tokenizer, prompts: list[str], device: str) -> np.ndarray:
    """Mean of normalized prompt embeddings, re-normalized."""
    embs = encode_prompts(model, tokenizer, prompts, device)
    mean = embs.mean(axis=0)
    return normalize_vectors(mean.reshape(1, -1)).squeeze(0).astype(np.float32)


def cosine_similarity_to_text(
    patch_features: np.ndarray,
    text_embedding: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between patch features (N, D) and one text vector (D,)."""
    pf = normalize_vectors(patch_features.astype(np.float64))
    te = normalize_vectors(text_embedding.reshape(1, -1).astype(np.float64)).squeeze(0)
    return (pf @ te).astype(np.float32)


def build_class_embeddings(
    class_names: list[str] | None = None,
    force: bool = False,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Step 8: build and save text prototype vectors per class."""
    out_dir = text_embeddings_dir()
    device = device or cfg.DEVICE
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"

    prompts_dict = cfg.TEXT_PROMPTS
    if class_names is None:
        class_names = list(prompts_dict.keys())

    if not force:
        all_exist = all((out_dir / f"{cn}.npy").exists() for cn in class_names if cn in prompts_dict)
        if all_exist:
            print("Text embeddings exist, loading from cache.")
            return {cn: np.load(out_dir / f"{cn}.npy") for cn in class_names if (out_dir / f"{cn}.npy").exists()}

    model, tokenizer = load_dinotxt(device)
    embeddings: dict[str, np.ndarray] = {}
    for cn in class_names:
        if cn not in prompts_dict:
            continue
        out_path = out_dir / f"{cn}.npy"
        if out_path.exists() and not force:
            embeddings[cn] = np.load(out_path)
            continue
        emb = ensemble_class_embedding(model, tokenizer, prompts_dict[cn], device)
        save_npy(out_path, emb)
        embeddings[cn] = emb
        print(f"  encoded '{cn}' from {len(prompts_dict[cn])} prompts")

    # Save class list manifest
    manifest_path = out_dir / "classes.txt"
    with open(manifest_path, "w") as f:
        f.write("\n".join(sorted(embeddings.keys())))

    print(f"Saved {len(embeddings)} class embeddings to {out_dir}")
    return embeddings


def load_class_embeddings() -> tuple[list[str], np.ndarray]:
    """Load all saved embeddings as (class_names, matrix (C, D))."""
    out_dir = text_embeddings_dir()
    manifest = out_dir / "classes.txt"
    if manifest.exists():
        class_names = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
    else:
        class_names = sorted(p.stem for p in out_dir.glob("*.npy"))
    embs = np.stack([np.load(out_dir / f"{cn}.npy") for cn in class_names])
    return class_names, embs.astype(np.float32)


def main():
    embs = build_class_embeddings(class_names=["chair", "table"], force=True, device="cpu")
    for cn, e in embs.items():
        print(f"{cn}: shape={e.shape}, norm={np.linalg.norm(e):.4f}")


if __name__ == "__main__":
    main()
