# -*- coding: utf-8 -*-
"""
gccm_algorithm.py
=================
Geographical Convergent Cross Mapping (GCCM) – Pure Python/NumPy Engine.

This module faithfully re-implements the R GCCM algorithm (see GCCM.r /
GCCM4Lattice.r in the package) in pure Python using NumPy and SciPy.
The computational logic is identical to the R version; differences are
limited to language idioms (0-based vs 1-based indexing, etc.).

Key functions
-------------
generate_grid_embeddings   Build spatial-lag embeddings for every pixel.
simplex_projection         Nearest-neighbour weighted cross-map prediction.
gccm_one_direction         Scan over library sizes for one causal direction.
gccm_bidirectional         Run both X→Y and Y→X and return combined results.
spatial_detrend            Remove linear spatial trend via OLS regression.
select_best_E              Grid-search over embedding dimensions.
"""

from __future__ import annotations

import warnings
import time
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


# ============================================================================
# Logging helpers  (10 % granularity)
# ============================================================================

def _print_stage(msg: str) -> None:
    print(f"[GCCM] {msg}", flush=True)


def _print_10pct_progress(prefix: str, i: int, n: int) -> None:
    """Print a progress line every 10 % of work completed.

    Parameters
    ----------
    prefix : str   Label shown in the log line.
    i      : int   Items completed so far (1-based).
    n      : int   Total number of items.
    """
    if n <= 0:
        return
    pct = int((i / n) * 100)
    if pct % 10 == 0:
        print(f"[{prefix}] {pct}% ({i}/{n})", flush=True)


# ============================================================================
# §0  Helpers
# ============================================================================

def r_seq(start: int, end: int) -> List[int]:
    """Return an integer range inclusive of *end* (R-style ``seq(start, end)``)."""
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, end - 1, -1))


def _sort_with_nan_last(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    nans = arr[~np.isfinite(arr)]
    return np.concatenate([np.sort(finite), nans])


# ============================================================================
# §1  Grid utilities
# ============================================================================

def _locate_grid_index(row: int, col: int, n_cols: int) -> int:
    return row * n_cols + col


def _rowcol_from_index(idx: int, n_cols: int) -> Tuple[int, int]:
    return divmod(idx, n_cols)


# ============================================================================
# §2  Spatial lag ring extraction
# ============================================================================

def _extract_lag_ring(mat: np.ndarray, row: int, col: int, lag: int) -> np.ndarray:
    """Return all values on the ring of distance *lag* around (row, col).

    Values that fall outside the matrix boundary are represented as NaN,
    matching the R implementation.
    """
    n_rows, n_cols = mat.shape
    ring_values: List[float] = []
    offsets = r_seq(lag, -lag)

    for dr in offsets:
        for dc in offsets:
            if abs(dr) != lag and abs(dc) != lag:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < n_rows and 0 <= c < n_cols:
                ring_values.append(mat[r, c])
            else:
                ring_values.append(np.nan)

    return np.array(ring_values, dtype=np.float64)


# ============================================================================
# §3  Embedding construction
# ============================================================================

def generate_grid_embeddings(
    mat: np.ndarray,
    E: int = 3,
    tau: int = 1,
) -> np.ndarray:
    """Construct spatial-lag state-space embeddings for every pixel.

    Parameters
    ----------
    mat : ndarray, shape (n_rows, n_cols)
        Input spatial field.
    E   : int
        Embedding dimension (number of spatial lags, including lag-0).
    tau : int
        Lag step size (ring radius increment).

    Returns
    -------
    embeddings : ndarray, shape (n_pixels, E)
        Row *i* is the embedding vector for pixel *i* in raster order.
        Dimension 0 is the pixel value itself; dimensions 1…E-1 are the
        ring-averaged values at successive spatial lags.
    """
    n_rows, n_cols = mat.shape
    n_pixels = n_rows * n_cols
    embeddings = np.full((n_pixels, E), np.nan, dtype=np.float64)

    flat = mat.ravel()
    _print_stage(f"Building embeddings: N={n_pixels}, E={E}, tau={tau}")

    for idx in range(n_pixels):
        _print_10pct_progress("embed", idx + 1, n_pixels)

        row, col = _rowcol_from_index(idx, n_cols)
        embeddings[idx, 0] = flat[idx]
        for e in range(1, E):
            lag = e * tau
            ring = _extract_lag_ring(mat, row, col, lag)
            embeddings[idx, e] = (
                np.nanmean(ring) if np.any(np.isfinite(ring)) else np.nan
            )

    _print_stage("Embedding construction complete")
    return embeddings


# ============================================================================
# §4  L1 distance matrix
# ============================================================================

def _compute_l1_distance_matrix_chunked(
    query: np.ndarray,
    library: np.ndarray,
    chunk_size: int = 500,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute the mean-L1 (Manhattan) distance matrix between query and library.

    NaN-safe: missing embedding dimensions are ignored when computing the
    per-row mean, exactly as ``rowMeans(…, na.rm=TRUE)`` in the R code.
    """
    n_q = query.shape[0]
    n_l = library.shape[0]
    dist = np.empty((n_q, n_l), dtype=np.float64)

    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        q_chunk = query[start:end]
        diff = np.abs(q_chunk[:, None, :] - library[None, :, :])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dist[start:end] = np.nanmean(diff, axis=2)

        if show_progress:
            _print_10pct_progress("dist", end, n_q)

    return dist


# ============================================================================
# §5  Simplex projection (cross-map prediction)
# ============================================================================

def simplex_projection(
    x_embeddings: np.ndarray,
    y_target: np.ndarray,
    lib_indices: np.ndarray,
    pred_indices: np.ndarray,
    b: int,
    chunk_size: int = 500,
) -> float:
    """Cross-map Y from the state space of X using simplex projection.

    For each prediction point, finds the *b* nearest neighbours in the
    library (excluding the point itself), computes exponential weights, and
    produces a weighted prediction of the corresponding Y value.  The final
    skill is the Pearson ρ between observed and predicted Y.

    Parameters
    ----------
    x_embeddings : ndarray, shape (n_pixels, E)
    y_target     : ndarray, shape (n_pixels,)
    lib_indices  : 1-D int array  — indices of library pixels
    pred_indices : 1-D int array  — indices of prediction pixels
    b            : int            — number of nearest neighbours
    chunk_size   : int            — chunk size for distance computation

    Returns
    -------
    rho : float  (NaN when fewer than 3 valid predictions are available)
    """
    lib_embeddings = x_embeddings[lib_indices]
    pred_embeddings = x_embeddings[pred_indices]

    dist_matrix = _compute_l1_distance_matrix_chunked(
        pred_embeddings, lib_embeddings,
        chunk_size=chunk_size, show_progress=False,
    )

    y_pred_list: List[float] = []
    y_obs_list: List[float] = []

    for i, p_idx in enumerate(pred_indices):
        dists = dist_matrix[i].copy()

        # exclude the point itself from the library
        self_mask = lib_indices == p_idx
        dists[self_mask] = np.inf

        if np.isfinite(dists).sum() < b:
            continue

        sorted_lib_order = np.argsort(dists)
        nn_idx = sorted_lib_order[:b]
        nn_dists = dists[nn_idx]

        min_dist = nn_dists[0]
        if min_dist == 0:
            weights = np.where(nn_dists == 0, 1.0, 0.0)
        else:
            weights = np.exp(-nn_dists / min_dist)

        w_sum = weights.sum()
        if w_sum == 0 or not np.isfinite(w_sum):
            continue
        weights /= w_sum

        nn_y = y_target[lib_indices[nn_idx]]
        if np.any(~np.isfinite(nn_y)):
            ok = np.isfinite(nn_y)
            if ok.sum() == 0:
                continue
            weights_ok = weights[ok] / weights[ok].sum()
            y_hat = np.dot(weights_ok, nn_y[ok])
        else:
            y_hat = np.dot(weights, nn_y)

        y_obs_val = y_target[p_idx]
        if not np.isfinite(y_obs_val):
            continue

        y_pred_list.append(y_hat)
        y_obs_list.append(y_obs_val)

    if len(y_pred_list) < 3:
        return np.nan

    rho, _ = stats.pearsonr(y_obs_list, y_pred_list)
    return float(rho)


# ============================================================================
# §6  Single direction for one library size
# ============================================================================

def _gccm_single(
    x_embeddings: np.ndarray,
    y_target: np.ndarray,
    lib_size: int,
    lib_indices: np.ndarray,
    pred_indices: np.ndarray,
    b: int,
    chunk_size: int = 500,
) -> List[float]:
    """Run simplex projection for all sliding-window positions of *lib_size*.

    When *lib_size* >= len(lib_indices), a single run over the full library
    is performed (no sliding window), matching the R behaviour.
    """
    max_lib = len(lib_indices)
    if lib_size >= max_lib:
        rho = simplex_projection(
            x_embeddings, y_target, lib_indices, pred_indices, b,
            chunk_size=chunk_size,
        )
        return [rho]

    rhos: List[float] = []
    _print_stage(f"Sliding-window rotation: lib_size={lib_size}, rotations={max_lib}")

    for start in range(max_lib):
        _print_10pct_progress(f"rotate(ls={lib_size})", start + 1, max_lib)

        if start + lib_size <= max_lib:
            local_lib = lib_indices[start : start + lib_size]
        else:
            local_lib = np.concatenate([
                lib_indices[start:],
                lib_indices[: lib_size - (max_lib - start)],
            ])

        rho = simplex_projection(
            x_embeddings, y_target, local_lib, pred_indices, b,
            chunk_size=chunk_size,
        )
        rhos.append(rho)

    _print_stage(f"Rotation complete: lib_size={lib_size}")
    return rhos


# ============================================================================
# §7  Evaluate one library size (used for parallel dispatch)
# ============================================================================

def _eval_one_libsize(
    ls: int,
    x_emb: np.ndarray,
    y_flat: np.ndarray,
    lib_indices: np.ndarray,
    pred_indices: np.ndarray,
    b: int,
    chunk_size: int,
) -> Tuple[int, float, float, float, float]:
    actual_ls = min(ls, len(lib_indices))
    rhos = _gccm_single(
        x_emb, y_flat, actual_ls, lib_indices, pred_indices, b,
        chunk_size=chunk_size,
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
# §8  Full GCCM for one causal direction
# ============================================================================

def gccm_one_direction(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    lib_sizes: List[int],
    E: int = 3,
    tau: int = 1,
    b: Optional[int] = None,
    chunk_size: int = 500,
    n_jobs: int = 1,
    precomputed_x_emb: Optional[np.ndarray] = None,
) -> dict:
    """Compute GCCM cross-map skill for all *lib_sizes* in one direction.

    Parameters
    ----------
    x_mat             : ndarray  — predictor spatial field
    y_mat             : ndarray  — target spatial field
    lib_sizes         : list[int] — library sizes to evaluate
    E                 : int      — embedding dimension
    tau               : int      — lag step
    b                 : int      — nearest-neighbour count (default E+1)
    chunk_size        : int      — distance-matrix chunk size
    n_jobs            : int      — parallel workers (1 = sequential)
    precomputed_x_emb : ndarray  — reuse pre-built embeddings if provided

    Returns
    -------
    dict with keys: lib_sizes, rho_mean, rho_sig, rho_lower, rho_upper
    """
    if b is None:
        b = E + 1

    x_emb = (
        precomputed_x_emb
        if precomputed_x_emb is not None
        else generate_grid_embeddings(x_mat, E=E, tau=tau)
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
    total_ls = len(lib_sizes)

    _print_stage(f"One-direction GCCM start: lib_sizes={lib_sizes}, n_jobs={n_jobs}")

    if n_jobs is None or n_jobs <= 1:
        for i, ls in enumerate(lib_sizes, start=1):
            _print_10pct_progress("one-dir(lib_sizes)", i, total_ls)
            _print_stage(f"lib_size={ls}")
            actual_ls, mean_rho, p_val, lower, upper = _eval_one_libsize(
                ls, x_emb, y_flat, lib_indices, pred_indices, b, chunk_size,
            )
            results["lib_sizes"].append(actual_ls)
            results["rho_mean"].append(mean_rho)
            results["rho_sig"].append(p_val)
            results["rho_lower"].append(lower)
            results["rho_upper"].append(upper)

        _print_stage("One-direction GCCM complete")
        return results

    # Parallel path
    _print_stage(f"Parallel mode: n_jobs={n_jobs}")
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(
                _eval_one_libsize, ls, x_emb, y_flat, lib_indices, pred_indices, b, chunk_size
            )
            for ls in lib_sizes
        ]
        lib_results = []
        for i, f in enumerate(futures, start=1):
            lib_results.append(f.result())
            _print_10pct_progress("one-dir(lib_sizes)", i, total_ls)

    for actual_ls, mean_rho, p_val, lower, upper in lib_results:
        results["lib_sizes"].append(actual_ls)
        results["rho_mean"].append(mean_rho)
        results["rho_sig"].append(p_val)
        results["rho_lower"].append(lower)
        results["rho_upper"].append(upper)

    _print_stage("One-direction GCCM complete")
    return results


# ============================================================================
# §9  Bidirectional GCCM
# ============================================================================

def gccm_bidirectional(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    lib_sizes: Optional[List[int]] = None,
    E: int = 3,
    tau: int = 1,
    b: Optional[int] = None,
    detrend: bool = True,
    chunk_size: int = 500,
    n_jobs: int = 1,
) -> dict:
    """Run GCCM in both X→Y and Y→X directions.

    Returns
    -------
    dict with keys ``x_xmap_y`` and ``y_xmap_x``, each being the output of
    :func:`gccm_one_direction`.
    """
    assert x_mat.shape == y_mat.shape, "x_mat and y_mat must have the same shape"

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

    _print_stage("Building X embedding …")
    t0 = time.time()
    x_emb = generate_grid_embeddings(x_mat, E=E, tau=tau)
    _print_stage(f"X embedding done ({time.time() - t0:.1f}s)")

    _print_stage("Building Y embedding …")
    t0 = time.time()
    y_emb = generate_grid_embeddings(y_mat, E=E, tau=tau)
    _print_stage(f"Y embedding done ({time.time() - t0:.1f}s)")

    _print_stage("Computing X xmap Y …")
    t0 = time.time()
    x_xmap_y = gccm_one_direction(
        x_mat, y_mat, lib_sizes, E=E, tau=tau, b=b,
        chunk_size=chunk_size, n_jobs=n_jobs, precomputed_x_emb=x_emb,
    )
    _print_stage(f"X xmap Y done ({time.time() - t0:.1f}s)")

    _print_stage("Computing Y xmap X …")
    t0 = time.time()
    y_xmap_x = gccm_one_direction(
        y_mat, x_mat, lib_sizes, E=E, tau=tau, b=b,
        chunk_size=chunk_size, n_jobs=n_jobs, precomputed_x_emb=y_emb,
    )
    _print_stage(f"Y xmap X done ({time.time() - t0:.1f}s)")

    return {"x_xmap_y": x_xmap_y, "y_xmap_x": y_xmap_x}


# ============================================================================
# §10  Spatial detrending
# ============================================================================

def spatial_detrend(mat: np.ndarray) -> np.ndarray:
    """Remove a first-order linear spatial trend via OLS regression.

    Fits ``value ~ col + row`` and returns the residuals.  Pixels that
    are NaN are excluded from the fit but kept as NaN in the output.
    """
    n_rows, n_cols = mat.shape
    rows, cols = np.meshgrid(
        np.arange(n_rows), np.arange(n_cols), indexing="ij"
    )
    rows_flat = rows.ravel().astype(np.float64)
    cols_flat = cols.ravel().astype(np.float64)
    vals_flat = mat.ravel().astype(np.float64)

    valid = np.isfinite(vals_flat)
    if valid.sum() < 3:
        return mat.copy()

    X_train = np.column_stack([cols_flat[valid], rows_flat[valid]])
    y_train = vals_flat[valid]
    reg = LinearRegression().fit(X_train, y_train)

    residuals = np.full_like(vals_flat, np.nan)
    X_all = np.column_stack([cols_flat, rows_flat])
    predicted = reg.predict(X_all)
    residuals[valid] = vals_flat[valid] - predicted[valid]

    return residuals.reshape(n_rows, n_cols)


# ============================================================================
# §11  E-selection
# ============================================================================

def select_best_E(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    E_range: Optional[List[int]] = None,
    tau: int = 1,
    detrend: bool = True,
    chunk_size: int = 500,
    n_jobs: int = 1,
) -> Tuple[int, dict]:
    """Select the best embedding dimension by cross-validation.

    Runs :func:`gccm_bidirectional` at ``lib_size = min(n_rows, n_cols)``
    for each candidate *E* and returns the one that maximises X→Y ρ.

    Returns
    -------
    best_E      : int
    all_results : dict  mapping each candidate E to its bidirectional result
    """
    if E_range is None:
        E_range = [2, 3, 4, 5]

    n_rows, n_cols = x_mat.shape
    lib_sizes = [min(n_rows, n_cols)]

    best_E = E_range[0]
    best_rho = -np.inf
    all_results: dict = {}

    _print_stage(f"E-search start: candidates={E_range}")

    for i, E in enumerate(E_range, start=1):
        _print_10pct_progress("E-search", i, len(E_range))
        _print_stage(f"Running E={E}")

        res = gccm_bidirectional(
            x_mat, y_mat,
            lib_sizes=lib_sizes, E=E, tau=tau, b=E + 1,
            detrend=detrend, chunk_size=chunk_size, n_jobs=n_jobs,
        )
        rho = res["x_xmap_y"]["rho_mean"][0]
        all_results[E] = res
        if np.isfinite(rho) and rho > best_rho:
            best_rho = rho
            best_E = E

    _print_stage(f"E-search done: best_E={best_E}, best_rho={best_rho:.6f}")
    return best_E, all_results
