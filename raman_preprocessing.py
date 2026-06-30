from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve
from sklearn.model_selection import train_test_split


def resolve_savgol_window(requested_window: int, seq_len: int) -> int:
    if int(requested_window) < 3:
        return 0
    window = int(requested_window)
    if window % 2 == 0:
        window += 1
    max_window = int(seq_len if seq_len % 2 == 1 else max(1, seq_len - 1))
    window = min(window, max_window)
    return window if window >= 3 else 0


def normalize_rows(X: np.ndarray, normalization_type: str = "minmax") -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim == 1:
        X_arr = X_arr[None, :]

    norm = str(normalization_type).strip().lower()
    if norm == "zscore":
        mean = X_arr.mean(axis=1, keepdims=True)
        std = np.maximum(X_arr.std(axis=1, keepdims=True), 1e-8)
        out = (X_arr - mean) / std
    else:
        x_min = X_arr.min(axis=1, keepdims=True)
        x_max = X_arr.max(axis=1, keepdims=True)
        out = (X_arr - x_min) / np.maximum(x_max - x_min, 1e-8)
    return out.astype(np.float32)


def _asls_baseline_1d(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float64)
    n = y_arr.shape[0]
    if n < 3:
        return y_arr.astype(np.float32)
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    penalty = float(max(1.0, lam)) * (d.T @ d)
    w = np.ones(n, dtype=np.float64)
    for _ in range(int(max(1, niter))):
        w_mat = sparse.spdiags(w, 0, n, n)
        z = spsolve(w_mat + penalty, w * y_arr)
        w = np.where(y_arr > z, float(np.clip(p, 1e-6, 1.0 - 1e-6)), 1.0 - float(np.clip(p, 1e-6, 1.0 - 1e-6)))
    return np.asarray(z, dtype=np.float32)


def _poly_baseline_1d(
    y: np.ndarray,
    order: int = 3,
    quantile: float = 0.35,
    niter: int = 3,
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float64)
    n = y_arr.shape[0]
    if n <= int(max(1, order)):
        return y_arr.astype(np.float32)
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    baseline_mask = y_arr <= np.quantile(y_arr, float(np.clip(quantile, 0.05, 0.95)))
    weights = np.where(baseline_mask, 1.0, 0.25).astype(np.float64)
    poly_order = int(min(max(1, order), n - 1))
    coeff = np.polyfit(x, y_arr, deg=poly_order, w=weights)
    baseline = np.polyval(coeff, x)
    for _ in range(int(max(1, niter)) - 1):
        resid = y_arr - baseline
        weights = np.where(resid <= 0.0, 1.0, 0.2).astype(np.float64)
        coeff = np.polyfit(x, y_arr, deg=poly_order, w=weights)
        baseline = np.polyval(coeff, x)
    return np.asarray(baseline, dtype=np.float32)


def baseline_correct_rows(
    X: np.ndarray,
    method: str = "als",
    als_lambda: float = 1e5,
    als_p: float = 0.01,
    als_niter: int = 10,
    poly_order: int = 3,
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim == 1:
        X_arr = X_arr[None, :]
    out = np.empty_like(X_arr, dtype=np.float32)
    meth = str(method).strip().lower()
    for idx, row in enumerate(X_arr):
        baseline = (
            _poly_baseline_1d(row, order=poly_order)
            if meth == "poly"
            else _asls_baseline_1d(row, lam=als_lambda, p=als_p, niter=als_niter)
        )
        corrected = np.asarray(row, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
        out[idx] = np.clip(corrected, a_min=0.0, a_max=None)
    return out


def preprocess_base_matrix(
    X: np.ndarray,
    *,
    use_log_scale: bool = True,
    use_baseline_correction: bool = False,
    baseline_method: str = "als",
    baseline_lambda: float = 1e5,
    baseline_p: float = 0.01,
    baseline_niter: int = 10,
    baseline_poly_order: int = 3,
    use_smoothing: bool = False,
    savgol_window: int = 0,
    savgol_poly: int = 2,
    normalization_type: str = "minmax",
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float32).copy()
    if X_arr.ndim == 1:
        X_arr = X_arr[None, :]
    if use_baseline_correction:
        X_arr = baseline_correct_rows(
            X_arr,
            method=baseline_method,
            als_lambda=baseline_lambda,
            als_p=baseline_p,
            als_niter=baseline_niter,
            poly_order=baseline_poly_order,
        )
    if use_log_scale:
        X_arr = np.log1p(np.clip(X_arr, 0.0, None)).astype(np.float32)
    if use_smoothing:
        window = resolve_savgol_window(savgol_window, X_arr.shape[1])
        if window > 0:
            poly = int(min(max(0, savgol_poly), window - 1))
            X_arr = savgol_filter(X_arr, window_length=window, polyorder=poly, axis=1).astype(np.float32)
    return normalize_rows(X_arr, normalization_type=normalization_type)


def build_feature_channels(
    base_matrix: np.ndarray,
    derivative_order: int = 0,
    normalization_type: str = "minmax",
) -> np.ndarray:
    base = np.asarray(base_matrix, dtype=np.float32)
    if base.ndim == 1:
        base = base[None, :]
    channels = [base]
    deriv_order = int(max(0, derivative_order))
    if deriv_order >= 1:
        d1 = np.gradient(base, axis=1).astype(np.float32)
        channels.append(normalize_rows(d1, normalization_type=normalization_type))
    if deriv_order >= 2:
        d2 = np.gradient(np.gradient(base, axis=1), axis=1).astype(np.float32)
        channels.append(normalize_rows(d2, normalization_type=normalization_type))
    if len(channels) == 1:
        return channels[0].astype(np.float32)
    return np.stack(channels, axis=1).astype(np.float32)


def preprocessing_config_signature(config: Dict[str, object]) -> str:
    text = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def export_preprocessed_merged_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    use_log_scale: bool = True,
    use_baseline_correction: bool = False,
    baseline_method: str = "als",
    baseline_lambda: float = 1e5,
    baseline_p: float = 0.01,
    baseline_niter: int = 10,
    baseline_poly_order: int = 3,
    use_smoothing: bool = False,
    savgol_window: int = 0,
    savgol_poly: int = 2,
    normalization_type: str = "minmax",
    use_derivative: bool = True,
    derivative_order: int = 0,
    holdout_fraction: float = 0.25,
    holdout_seed: int = 42,
    save_feature_channels: bool = True,
) -> Dict[str, object]:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(in_dir / "X.npy").astype(np.float32)
    y = np.load(in_dir / "y.npy").astype(np.int64)
    classes_map = json.loads((in_dir / "classes.json").read_text(encoding="utf-8"))

    base = preprocess_base_matrix(
        X,
        use_log_scale=use_log_scale,
        use_baseline_correction=use_baseline_correction,
        baseline_method=baseline_method,
        baseline_lambda=baseline_lambda,
        baseline_p=baseline_p,
        baseline_niter=baseline_niter,
        baseline_poly_order=baseline_poly_order,
        use_smoothing=use_smoothing,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        normalization_type=normalization_type,
    )
    deriv_order = int(max(0, derivative_order)) if bool(use_derivative) else 0

    feature_stack = build_feature_channels(
        base,
        derivative_order=deriv_order,
        normalization_type=normalization_type,
    ) if save_feature_channels else None

    holdout_frac = float(np.clip(holdout_fraction, 0.0, 0.5))
    if holdout_frac > 0.0:
        idx = np.arange(len(y), dtype=np.int64)
        train_idx, holdout_idx = train_test_split(
            idx,
            test_size=holdout_frac,
            stratify=y,
            random_state=int(holdout_seed),
        )
        train_idx = np.asarray(sorted(train_idx.tolist()), dtype=np.int64)
        holdout_idx = np.asarray(sorted(holdout_idx.tolist()), dtype=np.int64)
    else:
        train_idx = np.arange(len(y), dtype=np.int64)
        holdout_idx = np.empty((0,), dtype=np.int64)

    np.save(out_dir / "X.npy", base[train_idx].astype(np.float32))
    np.save(out_dir / "y.npy", y[train_idx].astype(np.int64))
    (out_dir / "classes.json").write_text(json.dumps(classes_map, indent=2), encoding="utf-8")

    if holdout_idx.size > 0:
        np.save(out_dir / "X_holdout.npy", base[holdout_idx].astype(np.float32))
        np.save(out_dir / "y_holdout.npy", y[holdout_idx].astype(np.int64))

    if feature_stack is not None:
        np.save(out_dir / "X_channels.npy", feature_stack[train_idx].astype(np.float32))
        if holdout_idx.size > 0:
            np.save(out_dir / "X_holdout_channels.npy", feature_stack[holdout_idx].astype(np.float32))

    config = {
        "source_dir": str(in_dir.resolve()),
        "use_log_scale": bool(use_log_scale),
        "use_baseline_correction": bool(use_baseline_correction),
        "baseline_method": str(baseline_method),
        "baseline_lambda": float(baseline_lambda),
        "baseline_p": float(baseline_p),
        "baseline_niter": int(baseline_niter),
        "baseline_poly_order": int(baseline_poly_order),
        "use_smoothing": bool(use_smoothing),
        "savgol_window": int(savgol_window),
        "savgol_poly": int(savgol_poly),
        "normalization_type": str(normalization_type),
        "use_derivative": bool(use_derivative),
        "derivative_order": int(deriv_order),
        "holdout_fraction": float(holdout_frac),
        "holdout_seed": int(holdout_seed),
        "save_feature_channels": bool(save_feature_channels),
        "n_train_samples": int(train_idx.size),
        "n_holdout_samples": int(holdout_idx.size),
        "seq_len": int(base.shape[1]),
        "signature": preprocessing_config_signature(
            {
                "use_log_scale": bool(use_log_scale),
                "use_baseline_correction": bool(use_baseline_correction),
                "baseline_method": str(baseline_method),
                "baseline_lambda": float(baseline_lambda),
                "baseline_p": float(baseline_p),
                "baseline_niter": int(baseline_niter),
                "baseline_poly_order": int(baseline_poly_order),
                "use_smoothing": bool(use_smoothing),
                "savgol_window": int(savgol_window),
                "savgol_poly": int(savgol_poly),
                "normalization_type": str(normalization_type),
                "use_derivative": bool(use_derivative),
                "derivative_order": int(deriv_order),
                "holdout_fraction": float(holdout_frac),
                "holdout_seed": int(holdout_seed),
            }
        ),
    }
    (out_dir / "preprocess_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config
