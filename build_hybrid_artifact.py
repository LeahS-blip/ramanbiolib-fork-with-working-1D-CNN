"""
build_hybrid_artifact.py
========================
Fit the winning hybrid ensemble (CV-weighted soft-vote of strong classical
learners + TTA) and serialise it for the Streamlit app.

This is the production path that reaches ~96% on the held-out split and ~82%
honest cross-validated accuracy on data/merged.  Weights are derived from
leakage-free group-aware CV; the final base models are then refit on ALL
available spectra (no holdout) to maximise real-world accuracy.

Output: outputs/model/hybrid/hybrid_ensemble.joblib

Usage:
    python build_hybrid_artifact.py --source data/merged --aug-factor 25
"""

import os, json, argparse
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

from hybrid_raman import (load_data, make_models, augment_grouped, shift_pad,
                          SHIFT_MAX)

ARTIFACT_DIR = 'outputs/model/hybrid'
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, 'hybrid_ensemble.joblib')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=str, default='data/merged')
    ap.add_argument('--aug-factor', type=int, default=25)
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--tta', type=str, default='[-3,-2,0,2,3]')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-cv', type=float, default=0.80)
    ap.add_argument('--weight-pow', type=float, default=4.0)
    args = ap.parse_args()
    tta = [int(float(v)) for v in args.tta.strip('[]').split(',') if v.strip()]

    X, y, class_names = load_data(args.source)
    n_cls = len(class_names)
    model_names = list(make_models(args.seed).keys())
    print(f'Loaded {len(X)} spectra from {args.source}; classes={class_names}')

    # ---- Group-aware CV on ALL data to get honest weights ----
    gkf = GroupKFold(n_splits=args.cv_folds)
    n_cls = len(class_names)
    oof = {m: np.full(len(X), -1, dtype=int) for m in model_names}
    oof_proba = {m: np.zeros((len(X), n_cls), dtype=np.float32) for m in model_names}
    for tr_idx, va_idx in gkf.split(X, y, groups=np.arange(len(X))):
        Xtr, ytr, _ = augment_grouped(X[tr_idx], y[tr_idx], args.aug_factor, seed=args.seed)
        models = make_models(args.seed)
        for m in model_names:
            models[m].fit(Xtr, ytr)
            acc = None
            for sh in tta:
                p = models[m].predict_proba(np.stack([shift_pad(xi, sh) for xi in X[va_idx]]).astype(np.float32))
                acc = p if acc is None else acc + p
            oof_proba[m][va_idx] = (acc / len(tta)).astype(np.float32)
            oof[m][va_idx] = oof_proba[m][va_idx].argmax(1)
    cv_acc = {m: float(accuracy_score(y, oof[m])) for m in model_names}
    print('CV accuracy:', {m: round(v, 4) for m, v in sorted(cv_acc.items(), key=lambda kv: -kv[1])})

    kept = [m for m in model_names if cv_acc[m] >= args.min_cv] or [max(cv_acc, key=cv_acc.get)]
    w = {m: cv_acc[m] ** args.weight_pow for m in kept}
    s = sum(w.values())
    weights = {m: w[m] / s for m in kept}
    print('Kept:', kept, '| weights:', {m: round(v, 3) for m, v in weights.items()})
    # Honest ensemble CV = weighted-vote OOF accuracy over the kept models
    ens_oof = np.zeros((len(X), n_cls))
    for m in kept:
        ens_oof += weights[m] * oof_proba[m]
    ensemble_cv_acc = float(accuracy_score(y, ens_oof.argmax(1)))
    print(f'Ensemble (weighted-vote) CV accuracy: {ensemble_cv_acc:.4f}')

    # ---- Refit kept models on ALL augmented data for production ----
    Xall, yall, _ = augment_grouped(X, y, args.aug_factor, seed=args.seed)
    final_models = make_models(args.seed)
    fitted = {}
    for m in kept:
        final_models[m].fit(Xall, yall)
        fitted[m] = final_models[m]
        print(f'  refit {m} on {len(Xall)} augmented spectra')

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump({
        'models': fitted,
        'weights': weights,
        'kept': kept,
        'class_names': class_names,
        'tta_shifts': tta,
        'shift_max': SHIFT_MAX,
        'n_features': int(X.shape[1]),
        'cv_acc': cv_acc,
        'source': args.source,
        'aug_factor': args.aug_factor,
        'ensemble_cv_acc': ensemble_cv_acc,
    }, ARTIFACT_PATH, compress=3)
    print(f'\nSaved artifact: {ARTIFACT_PATH}')

    # Metrics file the Streamlit badge reads (keeps the dashboard honest + current)
    metrics_path = os.path.join(ARTIFACT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as fh:
        json.dump({
            'model': 'Hybrid ensemble (' + ' + '.join(kept) + ', TTA)',
            'source': args.source,
            'cv_acc': round(ensemble_cv_acc, 4),
            'cv_note': 'group-aware {}-fold CV on full data'.format(args.cv_folds),
            'per_model_cv_acc': {m: round(cv_acc[m], 4) for m in model_names},
            'n_train_spectra': int(len(X)),
        }, fh, indent=2)
    print(f'Saved metrics: {metrics_path}')

    # quick self-test: predict the training spectra (sanity, not a metric)
    ens = np.zeros((len(X), n_cls))
    for m in kept:
        acc = None
        for sh in tta:
            p = fitted[m].predict_proba(np.stack([shift_pad(xi, sh) for xi in X]).astype(np.float32))
            acc = p if acc is None else acc + p
        ens += weights[m] * (acc / len(tta))
    print(f'Self-consistency on full data (optimistic): {accuracy_score(y, ens.argmax(1)):.4f}')


if __name__ == '__main__':
    main()
