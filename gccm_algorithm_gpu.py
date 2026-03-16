# -*- coding: utf-8 -*-
"""
gccm_algorithm_gpu.py
=====================
Geographical Convergent Cross Mapping (GCCM) – GPU-Accelerated Engine.

This module provides the same public API as ``gccm_algorithm.py`` but
accelerates the compute-intensive kernels with PyTorch.  When a CUDA-capable
GPU is detected it is used automatically; otherwise the code runs on the CPU
via PyTorch tensors (which is still faster than raw NumPy for large matrices
due to vectorisation).

All results are numerically identical to the CPU version within floating-point
rounding.

Accelerated operations
----------------------
* **Distance matrix** – the full (n_pred × n_lib) L1 distance matrix is
  computed in a single batched tensor operation on the GPU, eliminating
  the Python-level chunk loop.
* **Embedding construction** – ``generate_grid_embeddings_gpu`` uses
  ``torch.nn.functional.pad`` + 2-D convolution-style scatter to gather ring
  neighbours in one pass, avoiding the pixel-level Python loop.
* **Simplex projection** – k-NN selection and weighted prediction are
  performed entirely with ``torch.topk`` / batch matrix operations.

Usage
-----
    from gccm_algorithm_gpu import gccm_bidirectional_gpu

    result = gccm_bidirectional_gpu(x_mat, y_mat, lib_sizes=[10, 20, 30])

You can also import the drop-in replacements individually:

    from gccm_algorithm_gpu import (
        generate_grid_embeddings_gpu,
        simplex_projection_gpu,
        gccm_one_direction_gpu,
    )
"""

from __future__ import annotations

import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

# Shared helpers from the CPU engine
from gccm_algorithm import (
    _print_stage,
    _print_10pct_progress,
    spatial_detrend,
)

# Sentinel value used to replace non-finite distances before torch.topk,
# so that invalid entries sort to the end.
_DISTANCE_SENTINEL = 1e9


# ============================================================================
# Device selection
# ============================================================================

def _get_device(device: Optional[str] = None) -> "torch.device":
    """Return the best available torch device.

    Priority: ``device`` argument → CUDA → CPU.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed.  "
            "Install it with:  pip install torch"
        )
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info() -> str:
    """Return a human-readable string describing the active compute device."""
    if not _TORCH_AVAILABLE:
        return "PyTorch not available – falling back to NumPy CPU"
    dev = _get_device()
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(dev)
        return (
            f"GPU: {props.name}  "
            f"({props.total_memory / 1e9:.1f} GB, "
            f"CUDA {torch.version.cuda})"
        )
    return "CPU (torch)"


# ============================================================================
# §1  GPU embedding construction
# ============================================================================

def generate_grid_embeddings_gpu(
    mat: np.ndarray,
    E: int = 3,
    tau: int = 1,
    device: Optional[str] = None,
    batch_size: int = 2048,
) -> np.ndarray:
    """Build spatial-lag state-space embeddings using GPU tensor ops.

    This is a drop-in replacement for ``gccm_algorithm.generate_grid_embeddings``
    that avoids the Python pixel-level loop by:

    1. Padding the matrix with NaN borders.
    2. Using ``unfold`` to extract all (2·lag+1) × (2·lag+1) neighbourhoods
       in one call.
    3. Masking the inner region and taking the nanmean of the ring.

    Parameters
    ----------
    mat       : ndarray, shape (n_rows, n_cols)
    E         : int  — embedding dimension
    tau       : int  — lag step
    device    : str  — ``"cuda"``, ``"cpu"``, or None (auto-detect)
    batch_size: int  — number of pixels processed per GPU batch

    Returns
    -------
    embeddings : ndarray, shape (n_pixels, E)
    """
    dev = _get_device(device)
    n_rows, n_cols = mat.shape
    n_pixels = n_rows * n_cols

    _print_stage(
        f"GPU embedding: N={n_pixels}, E={E}, tau={tau}, device={dev}"
    )

    mat_t = torch.tensor(mat, dtype=torch.float32, device=dev)
    flat = mat_t.reshape(-1)  # (n_pixels,)

    embeddings = torch.full((n_pixels, E), float("nan"), dtype=torch.float32, device=dev)
    embeddings[:, 0] = flat  # lag-0: pixel value itself

    for e in range(1, E):
        lag = e * tau
        window = 2 * lag + 1
        # Pad with NaN on all four sides
        padded = F.pad(mat_t.unsqueeze(0).unsqueeze(0),
                       (lag, lag, lag, lag),
                       mode="constant",
                       value=float("nan"))  # (1, 1, n_rows+2*lag, n_cols+2*lag)

        # unfold: extract every window×window patch
        # result: (1, 1, n_rows, n_cols, window, window)
        patches = padded.unfold(2, window, 1).unfold(3, window, 1)
        patches = patches.squeeze(0).squeeze(0)          # (n_rows, n_cols, w, w)
        patches = patches.reshape(n_pixels, window, window)  # (n_pixels, w, w)

        # Build the ring mask: keep only cells at Chebyshev distance == lag
        r_idx = torch.arange(window, device=dev) - lag   # [-lag .. +lag]
        c_idx = torch.arange(window, device=dev) - lag
        dr, dc = torch.meshgrid(r_idx, c_idx, indexing="ij")
        ring_mask = (dr.abs() == lag) | (dc.abs() == lag)  # (w, w)  bool

        ring_vals = patches[:, ring_mask]  # (n_pixels, n_ring)

        # nanmean over the ring dimension
        valid = torch.isfinite(ring_vals)
        ring_sum = torch.where(valid, ring_vals, torch.zeros_like(ring_vals)).sum(dim=1)
        ring_cnt = valid.float().sum(dim=1)
        ring_mean = torch.where(ring_cnt > 0, ring_sum / ring_cnt, torch.full_like(ring_sum, float("nan")))

        embeddings[:, e] = ring_mean
        _print_10pct_progress("embed", e, E - 1)

    _print_stage("GPU embedding complete")
    return embeddings.cpu().numpy().astype(np.float64)


# ============================================================================
# §2  GPU distance matrix
# ============================================================================

def _compute_l1_distance_matrix_gpu(
    query: np.ndarray,
    library: np.ndarray,
    device: Optional[str] = None,
    chunk_size: int = 2000,
) -> np.ndarray:
    """Compute mean-L1 distance matrix using GPU tensor operations.

    NaN-safe: missing embedding dimensions are excluded from the mean,
    matching the NumPy (``np.nanmean``) and R (``rowMeans(…, na.rm=TRUE)``)
    behaviour.

    Parameters
    ----------
    query    : ndarray, shape (n_q, E)
    library  : ndarray, shape (n_l, E)
    device   : torch device string or None
    chunk_size: int  — query rows processed per GPU batch (tune to VRAM)

    Returns
    -------
    dist : ndarray, shape (n_q, n_l)
    """
    dev = _get_device(device)
    n_q = query.shape[0]
    n_l = library.shape[0]

    lib_t = torch.tensor(library, dtype=torch.float32, device=dev)  # (n_l, E)
    dist_out = np.empty((n_q, n_l), dtype=np.float32)

    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        q_chunk = torch.tensor(query[start:end], dtype=torch.float32, device=dev)

        # (chunk, n_l, E)
        diff = (q_chunk.unsqueeze(1) - lib_t.unsqueeze(0)).abs()
        valid = torch.isfinite(diff)

        diff_sum = torch.where(valid, diff, torch.zeros_like(diff)).sum(dim=2)
        diff_cnt = valid.float().sum(dim=2)
        chunk_dist = torch.where(
            diff_cnt > 0,
            diff_sum / diff_cnt,
            torch.full_like(diff_sum, float("nan")),
        )
        dist_out[start:end] = chunk_dist.cpu().numpy()

    return dist_out.astype(np.float64)


# ============================================================================
# §3  GPU simplex projection
# ============================================================================

def simplex_projection_gpu(
    x_embeddings: np.ndarray,
    y_target: np.ndarray,
    lib_indices: np.ndarray,
    pred_indices: np.ndarray,
    b: int,
    device: Optional[str] = None,
    chunk_size: int = 2000,
) -> float:
    """GPU-accelerated cross-map prediction (drop-in for ``simplex_projection``).

    The distance matrix is computed on the GPU; k-NN selection and weighted
    prediction are performed with batched torch operations.

    Returns
    -------
    rho : float  (NaN if fewer than 3 valid predictions)
    """
    dev = _get_device(device)

    lib_emb = x_embeddings[lib_indices]
    pred_emb = x_embeddings[pred_indices]

    # --- Distance matrix on GPU -----------------------------------------
    dist_np = _compute_l1_distance_matrix_gpu(
        pred_emb, lib_emb, device=str(dev), chunk_size=chunk_size
    )

    # --- kNN prediction on GPU ------------------------------------------
    dist_t = torch.tensor(dist_np, dtype=torch.float32, device=dev)  # (n_p, n_l)
    y_t = torch.tensor(y_target, dtype=torch.float32, device=dev)    # (n_pixels,)
    lib_t = torch.tensor(lib_indices, dtype=torch.long, device=dev)  # (n_l,)
    pred_t = torch.tensor(pred_indices, dtype=torch.long, device=dev)  # (n_p,)

    n_pred = len(pred_indices)

    # Mask out the point itself from the library
    lib_2d = lib_t.unsqueeze(0).expand(n_pred, -1)   # (n_p, n_l)
    pred_2d = pred_t.unsqueeze(1).expand(-1, len(lib_indices))  # (n_p, n_l)
    self_mask = lib_2d == pred_2d
    dist_t[self_mask] = float("inf")

    # Also set distances to inf where the library y-value is NaN
    lib_y = y_t[lib_t]  # (n_l,)
    nan_y_mask = ~torch.isfinite(lib_y)
    dist_t[:, nan_y_mask] = float("inf")

    # Check enough finite distances per prediction point
    finite_count = torch.isfinite(dist_t).sum(dim=1)  # (n_p,)
    enough = finite_count >= b

    # --- Gather kNNs with torch.topk (smallest distances) ---------------
    # Replace inf with a large sentinel for topk; we'll restore logic after
    dist_safe = dist_t.clone()
    dist_safe[~torch.isfinite(dist_safe)] = _DISTANCE_SENTINEL

    _, nn_idx = torch.topk(dist_safe, k=b, dim=1, largest=False)  # (n_p, b)
    nn_dists = dist_t.gather(1, nn_idx)  # (n_p, b)

    # Weights: exp(-d / min_d)
    min_dists = nn_dists[:, 0:1]  # (n_p, 1)

    zero_min = min_dists == 0
    weights = torch.where(
        zero_min.expand_as(nn_dists),
        (nn_dists == 0).float(),
        torch.exp(-nn_dists / min_dists.clamp(min=1e-12)),
    )

    # Zero-weight any non-finite nn_dists (shouldn't happen after sentinel, but safe)
    weights[~torch.isfinite(nn_dists)] = 0.0

    w_sum = weights.sum(dim=1, keepdim=True)  # (n_p, 1)
    valid_w = (w_sum > 0) & torch.isfinite(w_sum)
    weights = torch.where(
        valid_w.expand_as(weights),
        weights / w_sum.clamp(min=1e-12),
        torch.zeros_like(weights),
    )

    # Gather y-values of kNNs
    nn_lib_idx = lib_t[nn_idx]           # (n_p, b)  global pixel indices
    nn_y = y_t[nn_lib_idx]               # (n_p, b)

    # NaN-safe weighted sum
    nn_y_valid = torch.isfinite(nn_y)
    weights_safe = torch.where(nn_y_valid, weights, torch.zeros_like(weights))
    w_safe_sum = weights_safe.sum(dim=1, keepdim=True)
    valid_pred = (w_safe_sum > 0) & torch.isfinite(w_safe_sum) & enough.unsqueeze(1)
    weights_safe = torch.where(
        valid_pred.expand_as(weights_safe),
        weights_safe / w_safe_sum.clamp(min=1e-12),
        torch.zeros_like(weights_safe),
    )
    y_hat = (weights_safe * torch.where(nn_y_valid, nn_y, torch.zeros_like(nn_y))).sum(dim=1)

    # Observed y
    y_obs = y_t[pred_t]  # (n_p,)

    # Build valid mask: enough neighbours, valid weight sum, finite y_obs
    mask = (
        enough
        & valid_pred.squeeze(1)
        & torch.isfinite(y_obs)
        & torch.isfinite(y_hat)
    )

    y_pred_valid = y_hat[mask].cpu().numpy()
    y_obs_valid = y_obs[mask].cpu().numpy()

    if len(y_pred_valid) < 3:
        return np.nan

    rho, _ = stats.pearsonr(y_obs_valid, y_pred_valid)
    return float(rho)


# ============================================================================
# §4  GPU _gccm_single
# ============================================================================

def _gccm_single_gpu(
    x_embeddings: np.ndarray,
    y_target: np.ndarray,
    lib_size: int,
    lib_indices: np.ndarray,
    pred_indices: np.ndarray,
    b: int,
    device: Optional[str] = None,
    chunk_size: int = 2000,
) -> List[float]:
    """GPU-accelerated sliding-window GCCM for one library size."""
    max_lib = len(lib_indices)
    if lib_size >= max_lib:
        rho = simplex_projection_gpu(
            x_embeddings, y_target, lib_indices, pred_indices, b,
            device=device, chunk_size=chunk_size,
        )
        return [rho]

    rhos: List[float] = []
    _print_stage(f"GPU rotation: lib_size={lib_size}, rotations={max_lib}")

    for start in range(max_lib):
        _print_10pct_progress(f"rotate(ls={lib_size})", start + 1, max_lib)

        if start + lib_size <= max_lib:
            local_lib = lib_indices[start : start + lib_size]
        else:
            local_lib = np.concatenate([
                lib_indices[start:],
                lib_indices[: lib_size - (max_lib - start)],
            ])

        rho = simplex_projection_gpu(
            x_embeddings, y_target, local_lib, pred_indices, b,
            device=device, chunk_size=chunk_size,
        )
        rhos.append(rho)

    _print_stage(f"GPU rotation complete: lib_size={lib_size}")
    return rhos


# ============================================================================
# §5  GPU _eval_one_libsize
# ============================================================================

def _eval_one_libsize_gpu(
    ls: int,
    x_emb: np.ndarray,
    y_flat: np.ndarray,
    lib_indices: np.ndarray,
    pred_indices: np.ndarray,
    b: int,
    device: Optional[str] = None,
    chunk_size: int = 2000,
) -> Tuple[int, float, float, float, float]:
    actual_ls = min(ls, len(lib_indices))
    rhos = _gccm_single_gpu(
        x_emb, y_flat, actual_ls, lib_indices, pred_indices, b,
        device=device, chunk_size=chunk_size,
    )

    rhos_arr = np.array(rhos, dtype=np.float64)
    rhos_valid = rhos_arr[np.isfinite(rhos_arr)]

    if len(rhos_valid) == 0:
        return actual_ls, np.nan, np.nan, np.nan, np.nan

    mean_rho = float(np.nanmean(rhos_valid))

    if len(rhos_valid) > 1:
        _, p_val = stats.ttest_1samp(rhos_valid, 0)
        sem = float(np.nanstd(rhos_valid, ddof=1) / np.sqrt(len(rhos_valid)))
        lower = mean_rho - 1.96 * sem
        upper = mean_rho + 1.96 * sem
    else:
        p_val = np.nan
        lower = mean_rho
        upper = mean_rho

    return (
        actual_ls,
        mean_rho,
        float(p_val) if np.isfinite(p_val) else np.nan,
        float(lower),
        float(upper),
    )


# ============================================================================
# §6  GPU one-direction GCCM
# ============================================================================

def gccm_one_direction_gpu(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    lib_sizes: List[int],
    E: int = 3,
    tau: int = 1,
    b: Optional[int] = None,
    device: Optional[str] = None,
    chunk_size: int = 2000,
    precomputed_x_emb: Optional[np.ndarray] = None,
) -> dict:
    """GPU-accelerated GCCM for one causal direction.

    Drop-in replacement for ``gccm_algorithm.gccm_one_direction`` with an
    extra ``device`` parameter.

    Parameters
    ----------
    device : str or None
        ``"cuda"`` / ``"cuda:0"`` etc., ``"cpu"``, or None (auto-detect).
    chunk_size : int
        Number of query rows per GPU batch for the distance matrix.
        Reduce if you run out of GPU memory.
    """
    if not _TORCH_AVAILABLE:
        _print_stage("PyTorch not available – falling back to CPU NumPy engine")
        from gccm_algorithm import gccm_one_direction
        return gccm_one_direction(
            x_mat, y_mat, lib_sizes, E=E, tau=tau, b=b,
            chunk_size=chunk_size, n_jobs=1,
            precomputed_x_emb=precomputed_x_emb,
        )

    dev = _get_device(device)

    if b is None:
        b = E + 1

    x_emb = (
        precomputed_x_emb
        if precomputed_x_emb is not None
        else generate_grid_embeddings_gpu(x_mat, E=E, tau=tau, device=str(dev))
    )
    y_flat = y_mat.ravel().astype(np.float64)

    valid_mask = np.all(np.isfinite(x_emb), axis=1) & np.isfinite(y_flat)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < b + 1:
        return {
            "lib_sizes": lib_sizes,
            "rho_mean":  [np.nan] * len(lib_sizes),
            "rho_sig":   [np.nan] * len(lib_sizes),
            "rho_lower": [np.nan] * len(lib_sizes),
            "rho_upper": [np.nan] * len(lib_sizes),
        }

    lib_indices = valid_indices.copy()
    pred_indices = valid_indices.copy()
    results: dict = {
        "lib_sizes": [], "rho_mean": [], "rho_sig": [], "rho_lower": [], "rho_upper": []
    }

    _print_stage(
        f"GPU one-direction GCCM start: lib_sizes={lib_sizes}, device={dev}"
    )

    for i, ls in enumerate(lib_sizes, start=1):
        _print_10pct_progress("one-dir-gpu(lib_sizes)", i, len(lib_sizes))
        _print_stage(f"lib_size={ls}")

        actual_ls, mean_rho, p_val, lower, upper = _eval_one_libsize_gpu(
            ls, x_emb, y_flat, lib_indices, pred_indices, b,
            device=str(dev), chunk_size=chunk_size,
        )
        results["lib_sizes"].append(actual_ls)
        results["rho_mean"].append(mean_rho)
        results["rho_sig"].append(p_val)
        results["rho_lower"].append(lower)
        results["rho_upper"].append(upper)

    _print_stage("GPU one-direction GCCM complete")
    return results


# ============================================================================
# §7  GPU bidirectional GCCM
# ============================================================================

def gccm_bidirectional_gpu(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    lib_sizes: Optional[List[int]] = None,
    E: int = 3,
    tau: int = 1,
    b: Optional[int] = None,
    detrend: bool = True,
    device: Optional[str] = None,
    chunk_size: int = 2000,
) -> dict:
    """GPU-accelerated bidirectional GCCM.

    Drop-in replacement for ``gccm_algorithm.gccm_bidirectional`` that uses
    the GPU for all compute-intensive kernels.

    Parameters
    ----------
    x_mat, y_mat : ndarray, shape (n_rows, n_cols)
        Input spatial fields.
    lib_sizes    : list[int] or None  — library sizes to evaluate
    E            : int  — embedding dimension
    tau          : int  — lag step
    b            : int  — nearest-neighbour count (default E+1)
    detrend      : bool — remove linear spatial trend before analysis
    device       : str  — ``"cuda"``, ``"cpu"``, or None (auto-detect)
    chunk_size   : int  — GPU batch size for distance matrix computation

    Returns
    -------
    dict with keys ``x_xmap_y`` and ``y_xmap_x``.
    """
    assert x_mat.shape == y_mat.shape, "x_mat and y_mat must have the same shape"

    dev = _get_device(device)
    _print_stage(f"Device: {get_device_info()}")

    n_rows, n_cols = x_mat.shape

    if detrend:
        _print_stage("Spatial detrending …")
        t0 = time.time()
        x_mat = spatial_detrend(x_mat)
        y_mat = spatial_detrend(y_mat)
        _print_stage(f"Detrend complete ({time.time() - t0:.1f}s)")

    if lib_sizes is None:
        max_ls = min(n_rows, n_cols)
        lib_sizes = list(range(5, max_ls + 1, 5)) or [max_ls]

    _print_stage("Building X embedding (GPU) …")
    t0 = time.time()
    x_emb = generate_grid_embeddings_gpu(x_mat, E=E, tau=tau, device=str(dev))
    _print_stage(f"X embedding done ({time.time() - t0:.1f}s)")

    _print_stage("Building Y embedding (GPU) …")
    t0 = time.time()
    y_emb = generate_grid_embeddings_gpu(y_mat, E=E, tau=tau, device=str(dev))
    _print_stage(f"Y embedding done ({time.time() - t0:.1f}s)")

    _print_stage("Computing X xmap Y (GPU) …")
    t0 = time.time()
    x_xmap_y = gccm_one_direction_gpu(
        x_mat, y_mat, lib_sizes, E=E, tau=tau, b=b,
        device=str(dev), chunk_size=chunk_size,
        precomputed_x_emb=x_emb,
    )
    _print_stage(f"X xmap Y done ({time.time() - t0:.1f}s)")

    _print_stage("Computing Y xmap X (GPU) …")
    t0 = time.time()
    y_xmap_x = gccm_one_direction_gpu(
        y_mat, x_mat, lib_sizes, E=E, tau=tau, b=b,
        device=str(dev), chunk_size=chunk_size,
        precomputed_x_emb=y_emb,
    )
    _print_stage(f"Y xmap X done ({time.time() - t0:.1f}s)")

    return {"x_xmap_y": x_xmap_y, "y_xmap_x": y_xmap_x}


# ============================================================================
# §8  E-selection (GPU)
# ============================================================================

def select_best_E_gpu(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    E_range: Optional[List[int]] = None,
    tau: int = 1,
    detrend: bool = True,
    device: Optional[str] = None,
    chunk_size: int = 2000,
) -> Tuple[int, dict]:
    """GPU-accelerated E-selection (drop-in for ``gccm_algorithm.select_best_E``)."""
    if E_range is None:
        E_range = [2, 3, 4, 5]

    n_rows, n_cols = x_mat.shape
    lib_sizes = [min(n_rows, n_cols)]

    best_E = E_range[0]
    best_rho = -np.inf
    all_results: dict = {}

    _print_stage(f"E-search (GPU) start: candidates={E_range}")

    for i, E in enumerate(E_range, start=1):
        _print_10pct_progress("E-search-gpu", i, len(E_range))
        _print_stage(f"Running E={E}")

        res = gccm_bidirectional_gpu(
            x_mat, y_mat,
            lib_sizes=lib_sizes, E=E, tau=tau, b=E + 1,
            detrend=detrend, device=device, chunk_size=chunk_size,
        )
        rho = res["x_xmap_y"]["rho_mean"][0]
        all_results[E] = res
        if np.isfinite(rho) and rho > best_rho:
            best_rho = rho
            best_E = E

    _print_stage(f"E-search (GPU) done: best_E={best_E}, best_rho={best_rho:.6f}")
    return best_E, all_results
