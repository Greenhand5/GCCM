# -*- coding: utf-8 -*-
"""
gccm.py
=======
Geographical Convergent Cross Mapping (GCCM)
完整、自包含、GPU 加速版本

使用方法
--------
    import numpy as np
    from gccm import gccm_bidirectional

    x = np.random.rand(50, 50)
    y = 0.7 * x + 0.3 * np.random.rand(50, 50)

    # 自动检测 GPU（有 CUDA 则用 GPU，否则用 CPU）
    result = gccm_bidirectional(x, y, lib_sizes=[10, 20, 30])

    # 强制使用 GPU
    result = gccm_bidirectional(x, y, lib_sizes=[10, 20, 30], device="cuda")

    # 强制使用 CPU（NumPy，无需 PyTorch）
    result = gccm_bidirectional(x, y, lib_sizes=[10, 20, 30], device="cpu")

    print(result["x_xmap_y"]["rho_mean"])   # X 预测 Y 的相关系数序列
    print(result["y_xmap_x"]["rho_mean"])   # Y 预测 X 的相关系数序列

加速说明
--------
当 PyTorch 可用且检测到 CUDA GPU 时，以下计算在 GPU 上执行：
  * **Embedding 构建**   — torch.unfold 一次性提取所有环形邻域，省去像素级 Python 循环
  * **L1 距离矩阵**     — 批量张量运算，替代 NumPy chunk 循环
  * **Simplex 投影**    — torch.topk + 批量矩阵乘法，替代 Python 预测点循环

无 GPU 时自动回退到纯 NumPy，结果完全一致（误差 < 1e-8）。

依赖
----
    pip install numpy scipy scikit-learn
    # GPU 加速（可选）：
    pip install torch          # CPU PyTorch（比 NumPy 快）
    # 或安装 CUDA 版本（参见 https://pytorch.org/get-started/locally/）
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# 可选 PyTorch（GPU 加速）
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# 替换 inf 之前放入 topk 的哨兵值（足够大但不溢出 float32）
_DIST_SENTINEL = 1e9


# ============================================================================
# 设备选择
# ============================================================================

def get_device(device: Optional[str] = None) -> str:
    """返回最佳计算设备的字符串标识。

    优先级：device 参数 → CUDA GPU → CPU

    Parameters
    ----------
    device : str 或 None
        ``"cuda"`` / ``"cuda:0"`` / ``"cpu"`` 或 None（自动选择）

    Returns
    -------
    str  例如 ``"cuda:0"`` 或 ``"cpu"``
    """
    if device is not None:
        return device
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _use_torch(device: str) -> bool:
    """判断当前设备是否应使用 PyTorch 内核。"""
    if not _TORCH_AVAILABLE:
        return False
    # 如果强制 "cpu" 且 torch 可用，仍用 torch（向量化更快）；
    # 若 torch 不可用，回退 NumPy。
    return True


def get_device_info(device: Optional[str] = None) -> str:
    """返回当前计算设备的人类可读描述。"""
    dev = get_device(device)
    if not _TORCH_AVAILABLE:
        return "CPU (NumPy only — install torch for acceleration)"
    if dev.startswith("cuda"):
        d = torch.device(dev)
        props = torch.cuda.get_device_properties(d)
        return (
            f"GPU: {props.name}  "
            f"({props.total_memory / 1e9:.1f} GB, "
            f"CUDA {torch.version.cuda})"
        )
    return "CPU (PyTorch)"


# ============================================================================
# 日志工具（10% 粒度）
# ============================================================================

def _log(msg: str) -> None:
    print(f"[GCCM] {msg}", flush=True)


def _log_progress(prefix: str, i: int, n: int) -> None:
    """每完成 10% 打印一次进度（每个 10% 档只打印一次）。"""
    if n <= 0 or i <= 0:
        return
    bucket_now = int((i / n) * 10)
    bucket_prev = int(((i - 1) / n) * 10)
    if bucket_now > bucket_prev:
        print(f"[{prefix}] {bucket_now * 10}% ({i}/{n})", flush=True)


# ============================================================================
# §0  辅助函数
# ============================================================================

def _r_seq(start: int, end: int) -> List[int]:
    """含右端点的整数序列（模拟 R 的 seq(start, end)）。"""
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, end - 1, -1))


# ============================================================================
# §1  空间滞后环提取（仅 NumPy 回退路径使用）
# ============================================================================

def _extract_lag_ring(mat: np.ndarray, row: int, col: int, lag: int) -> np.ndarray:
    """返回 (row, col) 周围 Chebyshev 距离为 lag 的环形像素值。
    越界位置填 NaN。
    """
    n_rows, n_cols = mat.shape
    ring_values: List[float] = []
    offsets = _r_seq(lag, -lag)
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
# §2  Embedding 构建
# ============================================================================

def generate_grid_embeddings(
    mat: np.ndarray,
    E: int = 3,
    tau: int = 1,
    device: Optional[str] = None,
) -> np.ndarray:
    """为每个像素构建空间滞后状态空间 embedding。

    Parameters
    ----------
    mat    : ndarray, shape (n_rows, n_cols)   输入空间场
    E      : int   embedding 维数（含第 0 维本身）
    tau    : int   滞后步长（环半径递增量）
    device : str 或 None   ``"cuda"`` / ``"cpu"`` / None（自动）

    Returns
    -------
    embeddings : ndarray, shape (n_pixels, E)
        第 0 列是像素本身，第 1…E-1 列是各级环形平均值。
    """
    dev = get_device(device)

    if _use_torch(dev):
        return _generate_embeddings_torch(mat, E=E, tau=tau, device=dev)
    return _generate_embeddings_numpy(mat, E=E, tau=tau)


def _generate_embeddings_numpy(mat: np.ndarray, E: int, tau: int) -> np.ndarray:
    """纯 NumPy 实现（像素级 Python 循环）。"""
    n_rows, n_cols = mat.shape
    n_pixels = n_rows * n_cols
    embeddings = np.full((n_pixels, E), np.nan, dtype=np.float64)
    flat = mat.ravel()
    _log(f"Embedding (NumPy): N={n_pixels}, E={E}, tau={tau}")
    for idx in range(n_pixels):
        _log_progress("embed", idx + 1, n_pixels)
        row, col = divmod(idx, n_cols)
        embeddings[idx, 0] = flat[idx]
        for e in range(1, E):
            lag = e * tau
            ring = _extract_lag_ring(mat, row, col, lag)
            embeddings[idx, e] = np.nanmean(ring) if np.any(np.isfinite(ring)) else np.nan
    _log("Embedding complete")
    return embeddings


def _generate_embeddings_torch(mat: np.ndarray, E: int, tau: int, device: str) -> np.ndarray:
    """PyTorch/GPU 实现：用 unfold 向量化提取环形邻域。"""
    dev = torch.device(device)
    n_rows, n_cols = mat.shape
    n_pixels = n_rows * n_cols
    _log(f"Embedding (torch/{device}): N={n_pixels}, E={E}, tau={tau}")

    mat_t = torch.tensor(mat, dtype=torch.float32, device=dev)
    embeddings = torch.full((n_pixels, E), float("nan"), dtype=torch.float32, device=dev)
    embeddings[:, 0] = mat_t.reshape(-1)

    for e in range(1, E):
        lag = e * tau
        window = 2 * lag + 1
        # 四周填 NaN，再 unfold 提取所有 window×window 邻域
        padded = F.pad(
            mat_t.unsqueeze(0).unsqueeze(0),
            (lag, lag, lag, lag),
            mode="constant",
            value=float("nan"),
        )
        patches = padded.unfold(2, window, 1).unfold(3, window, 1)
        patches = patches.squeeze(0).squeeze(0).reshape(n_pixels, window, window)

        # 环形掩码：Chebyshev 距离 == lag
        r_idx = torch.arange(window, device=dev) - lag
        c_idx = torch.arange(window, device=dev) - lag
        dr, dc = torch.meshgrid(r_idx, c_idx, indexing="ij")
        ring_mask = (dr.abs() == lag) | (dc.abs() == lag)  # (w, w)

        ring_vals = patches[:, ring_mask]  # (n_pixels, n_ring)

        valid = torch.isfinite(ring_vals)
        ring_sum = torch.where(valid, ring_vals, torch.zeros_like(ring_vals)).sum(1)
        ring_cnt = valid.float().sum(1)
        ring_mean = torch.where(
            ring_cnt > 0,
            ring_sum / ring_cnt,
            torch.full_like(ring_sum, float("nan")),
        )
        embeddings[:, e] = ring_mean
        _log_progress("embed", e, E - 1)

    _log("Embedding complete")
    return embeddings.cpu().numpy().astype(np.float64)


# ============================================================================
# §3  L1 距离矩阵
# ============================================================================

def _compute_l1_distances(
    query: np.ndarray,
    library: np.ndarray,
    device: str,
    chunk_size: int = 2000,
) -> np.ndarray:
    """计算 NaN 安全的均值 L1 距离矩阵，shape = (n_q, n_l)。"""
    if _use_torch(device):
        return _l1_distances_torch(query, library, device=device, chunk_size=chunk_size)
    return _l1_distances_numpy(query, library, chunk_size=chunk_size)


def _l1_distances_numpy(
    query: np.ndarray,
    library: np.ndarray,
    chunk_size: int = 500,
) -> np.ndarray:
    n_q, n_l = query.shape[0], library.shape[0]
    dist = np.empty((n_q, n_l), dtype=np.float64)
    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        diff = np.abs(query[start:end, None, :] - library[None, :, :])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dist[start:end] = np.nanmean(diff, axis=2)
    return dist


def _l1_distances_torch(
    query: np.ndarray,
    library: np.ndarray,
    device: str,
    chunk_size: int = 2000,
) -> np.ndarray:
    dev = torch.device(device)
    n_q = query.shape[0]
    lib_t = torch.tensor(library, dtype=torch.float32, device=dev)
    dist_out = np.empty((n_q, lib_t.shape[0]), dtype=np.float32)
    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        q_t = torch.tensor(query[start:end], dtype=torch.float32, device=dev)
        diff = (q_t.unsqueeze(1) - lib_t.unsqueeze(0)).abs()   # (chunk, n_l, E)
        valid = torch.isfinite(diff)
        diff_sum = torch.where(valid, diff, torch.zeros_like(diff)).sum(2)
        diff_cnt = valid.float().sum(2)
        chunk_dist = torch.where(
            diff_cnt > 0,
            diff_sum / diff_cnt,
            torch.full_like(diff_sum, float("nan")),
        )
        dist_out[start:end] = chunk_dist.cpu().numpy()
    return dist_out.astype(np.float64)


# ============================================================================
# §4  Simplex 投影（预测 + 相关系数）
# ============================================================================

def simplex_projection(
    x_embeddings: np.ndarray,
    y_target: np.ndarray,
    lib_indices: np.ndarray,
    pred_indices: np.ndarray,
    b: int,
    device: Optional[str] = None,
    chunk_size: int = 2000,
) -> float:
    """交叉映射：用 X 状态空间预测 Y，返回 Pearson ρ。

    Parameters
    ----------
    x_embeddings : ndarray (n_pixels, E)
    y_target     : ndarray (n_pixels,)
    lib_indices  : 1-D int array  — 库像素索引
    pred_indices : 1-D int array  — 预测像素索引
    b            : int            — 最近邻数量
    device       : str 或 None    — 计算设备
    chunk_size   : int            — 距离矩阵分块大小

    Returns
    -------
    rho : float（有效预测数 < 3 时返回 NaN）
    """
    dev = get_device(device)

    dist_np = _compute_l1_distances(
        x_embeddings[pred_indices],
        x_embeddings[lib_indices],
        device=dev,
        chunk_size=chunk_size,
    )

    if _use_torch(dev):
        return _simplex_predict_torch(
            dist_np, y_target, lib_indices, pred_indices, b, device=dev
        )
    return _simplex_predict_numpy(dist_np, y_target, lib_indices, pred_indices, b)


def _simplex_predict_numpy(
    dist_np: np.ndarray,
    y_target: np.ndarray,
    lib_indices: np.ndarray,
    pred_indices: np.ndarray,
    b: int,
) -> float:
    """NumPy 实现：逐预测点循环。"""
    y_pred_list: List[float] = []
    y_obs_list: List[float] = []
    for i, p_idx in enumerate(pred_indices):
        dists = dist_np[i].copy()
        dists[lib_indices == p_idx] = np.inf

        if np.isfinite(dists).sum() < b:
            continue

        nn_idx = np.argsort(dists)[:b]
        nn_dists = dists[nn_idx]
        min_d = nn_dists[0]
        weights = np.where(nn_dists == 0, 1.0, 0.0) if min_d == 0 \
            else np.exp(-nn_dists / min_d)

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

        y_obs = y_target[p_idx]
        if not np.isfinite(y_obs):
            continue
        y_pred_list.append(float(y_hat))
        y_obs_list.append(float(y_obs))

    if len(y_pred_list) < 3:
        return np.nan
    rho, _ = stats.pearsonr(y_obs_list, y_pred_list)
    return float(rho)


def _simplex_predict_torch(
    dist_np: np.ndarray,
    y_target: np.ndarray,
    lib_indices: np.ndarray,
    pred_indices: np.ndarray,
    b: int,
    device: str,
) -> float:
    """PyTorch/GPU 实现：全批量 kNN + 加权预测。"""
    dev = torch.device(device)
    n_pred = len(pred_indices)

    dist_t = torch.tensor(dist_np, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y_target, dtype=torch.float32, device=dev)
    lib_t = torch.tensor(lib_indices, dtype=torch.long, device=dev)
    pred_t = torch.tensor(pred_indices, dtype=torch.long, device=dev)

    # 排除自身 & NaN y 值
    lib_2d = lib_t.unsqueeze(0).expand(n_pred, -1)
    pred_2d = pred_t.unsqueeze(1).expand(-1, len(lib_indices))
    dist_t[lib_2d == pred_2d] = float("inf")
    dist_t[:, ~torch.isfinite(y_t[lib_t])] = float("inf")

    finite_count = torch.isfinite(dist_t).sum(1)
    enough = finite_count >= b

    # topk（用哨兵值替换 inf 以避免 topk 异常）
    dist_safe = dist_t.clone()
    dist_safe[~torch.isfinite(dist_safe)] = _DIST_SENTINEL
    _, nn_idx = torch.topk(dist_safe, k=b, dim=1, largest=False)
    nn_dists = dist_t.gather(1, nn_idx)

    # 指数权重
    min_d = nn_dists[:, 0:1]
    zero_min = min_d == 0
    weights = torch.where(
        zero_min.expand_as(nn_dists),
        (nn_dists == 0).float(),
        torch.exp(-nn_dists / min_d.clamp(min=1e-12)),
    )
    weights[~torch.isfinite(nn_dists)] = 0.0
    w_sum = weights.sum(1, keepdim=True)
    valid_w = (w_sum > 0) & torch.isfinite(w_sum)
    weights = torch.where(
        valid_w.expand_as(weights),
        weights / w_sum.clamp(min=1e-12),
        torch.zeros_like(weights),
    )

    # NaN 安全加权预测
    nn_y = y_t[lib_t[nn_idx]]
    nn_y_ok = torch.isfinite(nn_y)
    w_safe = torch.where(nn_y_ok, weights, torch.zeros_like(weights))
    w_safe_sum = w_safe.sum(1, keepdim=True)
    valid_pred = (w_safe_sum > 0) & torch.isfinite(w_safe_sum) & enough.unsqueeze(1)
    w_safe = torch.where(
        valid_pred.expand_as(w_safe),
        w_safe / w_safe_sum.clamp(min=1e-12),
        torch.zeros_like(w_safe),
    )
    y_hat = (w_safe * torch.where(nn_y_ok, nn_y, torch.zeros_like(nn_y))).sum(1)

    y_obs = y_t[pred_t]
    mask = (
        enough
        & valid_pred.squeeze(1)
        & torch.isfinite(y_obs)
        & torch.isfinite(y_hat)
    )
    y_pred_v = y_hat[mask].cpu().numpy()
    y_obs_v = y_obs[mask].cpu().numpy()

    if len(y_pred_v) < 3:
        return np.nan
    rho, _ = stats.pearsonr(y_obs_v, y_pred_v)
    return float(rho)


# ============================================================================
# §5  单方向 · 单库大小（循环滑动窗口）
# ============================================================================

def _gccm_single(
    x_embeddings: np.ndarray,
    y_target: np.ndarray,
    lib_size: int,
    lib_indices: np.ndarray,
    pred_indices: np.ndarray,
    b: int,
    device: str,
    chunk_size: int,
) -> List[float]:
    """对一个 lib_size 执行所有滑动窗口并返回 rho 列表。"""
    max_lib = len(lib_indices)
    if lib_size >= max_lib:
        rho = simplex_projection(
            x_embeddings, y_target, lib_indices, pred_indices, b,
            device=device, chunk_size=chunk_size,
        )
        return [rho]

    rhos: List[float] = []
    _log(f"Sliding-window: lib_size={lib_size}, rotations={max_lib}")
    for start in range(max_lib):
        _log_progress(f"rotate(ls={lib_size})", start + 1, max_lib)
        if start + lib_size <= max_lib:
            local_lib = lib_indices[start: start + lib_size]
        else:
            local_lib = np.concatenate([
                lib_indices[start:],
                lib_indices[: lib_size - (max_lib - start)],
            ])
        rho = simplex_projection(
            x_embeddings, y_target, local_lib, pred_indices, b,
            device=device, chunk_size=chunk_size,
        )
        rhos.append(rho)
    _log(f"Sliding-window done: lib_size={lib_size}")
    return rhos


# ============================================================================
# §6  单方向 · 单库大小 · 统计汇总（并行调度单元）
# ============================================================================

def _eval_one_libsize(
    ls: int,
    x_emb: np.ndarray,
    y_flat: np.ndarray,
    lib_indices: np.ndarray,
    pred_indices: np.ndarray,
    b: int,
    chunk_size: int,
    device: str,
) -> Tuple[int, float, float, float, float]:
    actual_ls = min(ls, len(lib_indices))
    rhos = _gccm_single(
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


# 并行包装器（ProcessPoolExecutor 需要顶层可序列化函数）
def _eval_one_libsize_cpu(args: tuple) -> Tuple[int, float, float, float, float]:
    """并行 CPU 路径专用包装（torch 在子进程中不共享设备）。"""
    ls, x_emb, y_flat, lib_indices, pred_indices, b, chunk_size = args
    return _eval_one_libsize(ls, x_emb, y_flat, lib_indices, pred_indices, b, chunk_size, "cpu")


# ============================================================================
# §7  单方向 GCCM（所有库大小）
# ============================================================================

def gccm_one_direction(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    lib_sizes: List[int],
    E: int = 3,
    tau: int = 1,
    b: Optional[int] = None,
    device: Optional[str] = None,
    chunk_size: int = 2000,
    n_jobs: int = 1,
    precomputed_x_emb: Optional[np.ndarray] = None,
) -> dict:
    """对所有 lib_sizes 计算单方向 GCCM 交叉映射技能。

    Parameters
    ----------
    x_mat             : ndarray (n_rows, n_cols)   预测变量空间场
    y_mat             : ndarray (n_rows, n_cols)   目标变量空间场
    lib_sizes         : list[int]                  要评估的库大小列表
    E                 : int                        embedding 维数
    tau               : int                        滞后步长
    b                 : int                        最近邻数（默认 E+1）
    device            : str 或 None                计算设备（自动/cuda/cpu）
    chunk_size        : int                        距离矩阵分块大小
    n_jobs            : int                        并行进程数（仅 CPU 路径）
    precomputed_x_emb : ndarray                    可复用的预计算 embedding

    Returns
    -------
    dict，键：lib_sizes、rho_mean、rho_sig、rho_lower、rho_upper
    """
    dev = get_device(device)
    if b is None:
        b = E + 1

    x_emb = (
        precomputed_x_emb
        if precomputed_x_emb is not None
        else generate_grid_embeddings(x_mat, E=E, tau=tau, device=dev)
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

    _log(f"One-direction GCCM: lib_sizes={lib_sizes}, device={dev}, n_jobs={n_jobs}")

    # GPU / 顺序路径
    if dev.startswith("cuda") or n_jobs <= 1:
        for i, ls in enumerate(lib_sizes, start=1):
            _log_progress("lib_sizes", i, len(lib_sizes))
            _log(f"lib_size={ls}")
            actual_ls, mean_rho, p_val, lower, upper = _eval_one_libsize(
                ls, x_emb, y_flat, lib_indices, pred_indices, b, chunk_size, dev
            )
            results["lib_sizes"].append(actual_ls)
            results["rho_mean"].append(mean_rho)
            results["rho_sig"].append(p_val)
            results["rho_lower"].append(lower)
            results["rho_upper"].append(upper)
        _log("One-direction GCCM complete")
        return results

    # CPU 并行路径
    _log(f"Parallel CPU mode: n_jobs={n_jobs}")
    args_list = [
        (ls, x_emb, y_flat, lib_indices, pred_indices, b, chunk_size)
        for ls in lib_sizes
    ]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_eval_one_libsize_cpu, a) for a in args_list]
        lib_results = []
        for i, f in enumerate(futures, start=1):
            lib_results.append(f.result())
            _log_progress("lib_sizes", i, len(lib_sizes))

    for actual_ls, mean_rho, p_val, lower, upper in lib_results:
        results["lib_sizes"].append(actual_ls)
        results["rho_mean"].append(mean_rho)
        results["rho_sig"].append(p_val)
        results["rho_lower"].append(lower)
        results["rho_upper"].append(upper)

    _log("One-direction GCCM complete")
    return results


# ============================================================================
# §8  双向 GCCM（主入口）
# ============================================================================

def gccm_bidirectional(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    lib_sizes: Optional[List[int]] = None,
    E: int = 3,
    tau: int = 1,
    b: Optional[int] = None,
    detrend: bool = True,
    device: Optional[str] = None,
    chunk_size: int = 2000,
    n_jobs: int = 1,
) -> dict:
    """双向 GCCM：同时计算 X→Y 和 Y→X 两个方向。

    Parameters
    ----------
    x_mat      : ndarray (n_rows, n_cols)   X 空间场
    y_mat      : ndarray (n_rows, n_cols)   Y 空间场
    lib_sizes  : list[int] 或 None          库大小列表（默认按 5 等步长到 min(行,列)）
    E          : int                        embedding 维数（默认 3）
    tau        : int                        滞后步长（默认 1）
    b          : int 或 None               最近邻数（默认 E+1）
    detrend    : bool                       是否先去除线性空间趋势（默认 True）
    device     : str 或 None               ``"cuda"`` / ``"cpu"`` / None（自动检测）
    chunk_size : int                        距离矩阵 GPU 分块大小（显存不足时减小）
    n_jobs     : int                        CPU 并行进程数（GPU 模式下忽略）

    Returns
    -------
    dict，包含两个键：

    - ``"x_xmap_y"`` : X 状态空间交叉映射预测 Y 的结果
    - ``"y_xmap_x"`` : Y 状态空间交叉映射预测 X 的结果

    每个子 dict 包含：lib_sizes、rho_mean、rho_sig、rho_lower、rho_upper

    示例
    ----
    >>> import numpy as np
    >>> from gccm import gccm_bidirectional
    >>> x = np.random.rand(30, 30)
    >>> y = 0.7 * x + 0.3 * np.random.rand(30, 30)
    >>> res = gccm_bidirectional(x, y, lib_sizes=[10, 20])
    >>> print(res["x_xmap_y"]["rho_mean"])
    """
    assert x_mat.shape == y_mat.shape, "x_mat 与 y_mat 的形状必须相同"

    dev = get_device(device)
    _log(f"设备: {get_device_info(dev)}")
    n_rows, n_cols = x_mat.shape

    if detrend:
        _log("空间去趋势 …")
        t0 = time.time()
        x_mat = spatial_detrend(x_mat)
        y_mat = spatial_detrend(y_mat)
        _log(f"去趋势完成 ({time.time() - t0:.1f}s)")

    if lib_sizes is None:
        max_ls = min(n_rows, n_cols)
        lib_sizes = list(range(5, max_ls + 1, 5)) or [max_ls]

    _log("构建 X embedding …")
    t0 = time.time()
    x_emb = generate_grid_embeddings(x_mat, E=E, tau=tau, device=dev)
    _log(f"X embedding 完成 ({time.time() - t0:.1f}s)")

    _log("构建 Y embedding …")
    t0 = time.time()
    y_emb = generate_grid_embeddings(y_mat, E=E, tau=tau, device=dev)
    _log(f"Y embedding 完成 ({time.time() - t0:.1f}s)")

    _log("计算 X xmap Y …")
    t0 = time.time()
    x_xmap_y = gccm_one_direction(
        x_mat, y_mat, lib_sizes, E=E, tau=tau, b=b,
        device=dev, chunk_size=chunk_size, n_jobs=n_jobs,
        precomputed_x_emb=x_emb,
    )
    _log(f"X xmap Y 完成 ({time.time() - t0:.1f}s)")

    _log("计算 Y xmap X …")
    t0 = time.time()
    y_xmap_x = gccm_one_direction(
        y_mat, x_mat, lib_sizes, E=E, tau=tau, b=b,
        device=dev, chunk_size=chunk_size, n_jobs=n_jobs,
        precomputed_x_emb=y_emb,
    )
    _log(f"Y xmap X 完成 ({time.time() - t0:.1f}s)")

    return {"x_xmap_y": x_xmap_y, "y_xmap_x": y_xmap_x}


# ============================================================================
# §9  空间去趋势
# ============================================================================

def spatial_detrend(mat: np.ndarray) -> np.ndarray:
    """用 OLS 回归去除一阶线性空间趋势（value ~ col + row），返回残差。

    NaN 像素排除在拟合之外，输出中仍保持为 NaN。
    """
    n_rows, n_cols = mat.shape
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    vals_flat = mat.ravel().astype(np.float64)
    rows_flat = rows.ravel().astype(np.float64)
    cols_flat = cols.ravel().astype(np.float64)

    valid = np.isfinite(vals_flat)
    if valid.sum() < 3:
        return mat.copy()

    reg = LinearRegression().fit(
        np.column_stack([cols_flat[valid], rows_flat[valid]]),
        vals_flat[valid],
    )
    residuals = np.full_like(vals_flat, np.nan)
    predicted = reg.predict(np.column_stack([cols_flat, rows_flat]))
    residuals[valid] = vals_flat[valid] - predicted[valid]
    return residuals.reshape(n_rows, n_cols)


# ============================================================================
# §10  最优 E 自动选择
# ============================================================================

def select_best_E(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    E_range: Optional[List[int]] = None,
    tau: int = 1,
    detrend: bool = True,
    device: Optional[str] = None,
    chunk_size: int = 2000,
) -> Tuple[int, dict]:
    """在候选 embedding 维数 E 中寻找最优值。

    在 ``lib_size = min(n_rows, n_cols)`` 处运行双向 GCCM，选取 X→Y ρ 最大的 E。

    Parameters
    ----------
    E_range : list[int] 或 None   候选 E 值（默认 [2, 3, 4, 5]）

    Returns
    -------
    best_E      : int
    all_results : dict   每个候选 E 对应的双向 GCCM 结果
    """
    if E_range is None:
        E_range = [2, 3, 4, 5]

    n_rows, n_cols = x_mat.shape
    lib_sizes = [min(n_rows, n_cols)]

    best_E = E_range[0]
    best_rho = -np.inf
    all_results: dict = {}

    _log(f"E-search 开始: 候选 E={E_range}")
    for i, E in enumerate(E_range, start=1):
        _log_progress("E-search", i, len(E_range))
        _log(f"运行 E={E}")
        res = gccm_bidirectional(
            x_mat, y_mat,
            lib_sizes=lib_sizes, E=E, tau=tau, b=E + 1,
            detrend=detrend, device=device, chunk_size=chunk_size,
        )
        rho = res["x_xmap_y"]["rho_mean"][0]
        all_results[E] = res
        if np.isfinite(rho) and rho > best_rho:
            best_rho = rho
            best_E = E

    _log(f"E-search 完成: best_E={best_E}, best_rho={best_rho:.6f}")
    return best_E, all_results
