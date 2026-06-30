"""
hybrid_raman.py
==============
Principled, leakage-free ensemble for the single-label Raman classifier.

Lessons that shaped this design (see outputs/logs/*.json):
  * A 1.2M-param CNN is high-variance on 123 spectra (test 73-86% by seed);
    a Random Forest already reaches ~88% and is far more stable.
  * Across model FAMILIES only test samples #7 and #42 are unrecoverable by
    any model, so the ceiling on this holdout is ~95.9% -- but the gains must
    come from choices validated by cross-validation, NOT by peeking at the
    49-sample test set.

What this script does:
  1. Identical holdout to train_cnn_raman.py (split seed=42, 25% test).
  2. GROUP-aware cross-validation on the 145 training spectra: augmented copies
     of a spectrum never straddle the train/val fold boundary (no leakage), so
     the CV accuracy is an honest estimate of generalisation.
  3. Each base learner's weight in the soft-vote = its CV accuracy (weak models
     are down-weighted automatically; below --min-cv they are dropped).
  4. Test-time augmentation (small spectral shifts) at prediction.
  5. Reports CV accuracy (the trustworthy number) AND the holdout accuracy.

Usage:
    python hybrid_raman.py
    python hybrid_raman.py --aug-factor 25 --cv-folds 5 --tta "[-3,-2,0,2,3]"
"""

import os, json, argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

SPLIT_SEED = 42
KEEP_CLASSES = ['Proteins', 'Lipids', 'Saccharides', 'AminoAcids', 'PrimaryMetabolites', 'NucleicAcids']
SHIFT_MAX, SCALE_MIN, SCALE_MAX = 4, 0.92, 1.08
GAUSSIAN_NOISE_STD, POISSON_NOISE_SCALE = 0.01, 0.008
BASELINE_WARP_SCALE, BROADENING_SIGMA_MAX = 0.03, 1.0


def shift_pad(x, s):
    if s == 0:
        return x.copy()
    o = np.empty_like(x)
    if s > 0:
        o[:s] = x[0]; o[s:] = x[:-s]
    else:
        k = -s; o[-k:] = x[-1]; o[:-k] = x[k:]
    return o


def augment_grouped(X, y, factor, seed):
    """Return augmented X, y, and group ids (group = index of source spectrum)."""
    axis = np.linspace(-1, 1, X.shape[1], dtype=np.float32)
    groups0 = np.arange(len(X))
    Xa, ya, ga = [X.copy()], [y.copy()], [groups0.copy()]
    rng = np.random.default_rng(seed)
    for _ in range(factor):
        Xn = np.empty_like(X)
        for i, sp in enumerate(X):
            a = shift_pad(sp.copy(), int(rng.integers(-SHIFT_MAX, SHIFT_MAX + 1)))
            sig = float(rng.uniform(0, BROADENING_SIGMA_MAX))
            if sig > 1e-6:
                a = gaussian_filter1d(a, sig, mode='nearest')
            a = a * float(rng.uniform(SCALE_MIN, SCALE_MAX))
            a = a + (rng.normal(0, BASELINE_WARP_SCALE * 0.35)
                     + rng.normal(0, BASELINE_WARP_SCALE * 0.55) * axis
                     + rng.normal(0, BASELINE_WARP_SCALE * 0.55) * (axis ** 2 - 0.33)).astype(np.float32)
            a = a + rng.normal(0, POISSON_NOISE_SCALE * np.sqrt(np.clip(a, 0, None) + 1e-6), a.shape).astype(np.float32)
            a = a + rng.normal(0, GAUSSIAN_NOISE_STD, a.shape).astype(np.float32)
            Xn[i] = np.clip(a, 0, None)
        Xa.append(Xn); ya.append(y.copy()); ga.append(groups0.copy())
    return np.vstack(Xa), np.concatenate(ya), np.concatenate(ga)


def make_models(seed):
    return {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=5000, C=1.0, class_weight='balanced', solver='lbfgs')),
        'svm_rbf': make_pipeline(StandardScaler(), SVC(
            C=10, gamma='scale', class_weight='balanced', probability=True, random_state=seed)),
        'svm_lin': make_pipeline(StandardScaler(), SVC(
            C=1, kernel='linear', class_weight='balanced', probability=True, random_state=seed)),
        'rf': RandomForestClassifier(n_estimators=400, random_state=seed,
                                     class_weight='balanced_subsample', n_jobs=-1),
        'extratrees': ExtraTreesClassifier(n_estimators=400, random_state=seed,
                                           class_weight='balanced_subsample', n_jobs=-1),
        'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance')),
    }


def load_data(source='ramanbiolib'):
    if source != 'ramanbiolib':
        # Load a pre-built merged dataset (X.npy / y.npy / classes.json)
        X = np.load(os.path.join(source, 'X.npy')).astype(np.float32)
        y_raw = np.load(os.path.join(source, 'y.npy'))
        with open(os.path.join(source, 'classes.json')) as fh:
            cls_map = json.load(fh)             # {name: idx}
        idx_to_name = {v: k for k, v in cls_map.items()}
        names_in_order = [idx_to_name[i] for i in sorted(idx_to_name)]
        # Re-encode to alphabetical class order for consistency with the CNN script
        le = LabelEncoder(); le.fit(KEEP_CLASSES)
        y = le.transform([names_in_order[int(v)] for v in y_raw])
        # per-spectrum min-max normalise (match ramanbiolib convention)
        X = (X - X.min(1, keepdims=True)) / (X.max(1, keepdims=True) - X.min(1, keepdims=True) + 1e-9)
        return X.astype(np.float32), y, list(le.classes_)
    def pl(s):
        return [float(v) for v in s.strip('[]').split(', ')]
    sp = pd.read_csv('ramanbiolib/db/raman_spectra_db.csv', converters={'wavenumbers': pl, 'intensity': pl})
    md = pd.read_csv('ramanbiolib/db/metadata_db.csv')
    df = sp.merge(md[['id', 'type']].drop_duplicates('id'), on='id')
    df['class'] = df['type'].str.split('/').str[0]
    df = df[df['class'].isin(KEEP_CLASSES)].reset_index(drop=True)
    X = np.array(df['intensity'].tolist(), dtype=np.float32)
    le = LabelEncoder(); y = le.fit_transform(df['class'])
    return X, y, list(le.classes_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aug-factor', type=int, default=25)
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--tta', type=str, default='[-3,-2,0,2,3]')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-cv', type=float, default=0.80, help='Drop base models with CV acc below this')
    ap.add_argument('--weight-pow', type=float, default=4.0, help='Soft-vote weight = cv_acc ** weight_pow')
    ap.add_argument('--source', type=str, default='data/merged',
                    help="'ramanbiolib' or a merged-dataset dir (e.g. data/merged, data/merged_relaxed_20260406)")
    ap.add_argument('--out', type=str, default='outputs/logs/hybrid_results.json')
    args = ap.parse_args()
    tta = [int(float(v)) for v in args.tta.strip('[]').split(',') if v.strip()]

    X, y, class_names = load_data(args.source)
    n_cls = len(class_names)
    X_tr_raw, X_test, y_tr_raw, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SPLIT_SEED)
    print(f'Train spectra {len(X_tr_raw)}  Test {len(X_test)}  aug={args.aug_factor}  cv={args.cv_folds}  tta={tta}')

    model_names = list(make_models(args.seed).keys())

    # ---- Group-aware CV to estimate each model's honest accuracy ----
    gkf = GroupKFold(n_splits=args.cv_folds)
    oof_pred = {m: np.full(len(X_tr_raw), -1, dtype=int) for m in model_names}
    oof_proba = {m: np.zeros((len(X_tr_raw), n_cls), dtype=np.float32) for m in model_names}
    for tr_idx, va_idx in gkf.split(X_tr_raw, y_tr_raw, groups=np.arange(len(X_tr_raw))):
        Xtr, ytr, _ = augment_grouped(X_tr_raw[tr_idx], y_tr_raw[tr_idx], args.aug_factor, seed=args.seed)
        Xva = X_tr_raw[va_idx]
        models = make_models(args.seed)
        for m in model_names:
            models[m].fit(Xtr, ytr)
            # TTA on the held-out originals
            acc = None
            for sh in tta:
                p = models[m].predict_proba(np.stack([shift_pad(xi, sh) for xi in Xva]).astype(np.float32))
                acc = p if acc is None else acc + p
            oof_proba[m][va_idx] = (acc / len(tta)).astype(np.float32)
            oof_pred[m][va_idx] = oof_proba[m][va_idx].argmax(1)

    cv_acc = {m: accuracy_score(y_tr_raw, oof_pred[m]) for m in model_names}
    print('\nGroup-aware CV accuracy (honest generalisation estimate):')
    for m in sorted(cv_acc, key=cv_acc.get, reverse=True):
        print(f'  {m:11s} cv_acc={cv_acc[m]:.4f}')

    kept = [m for m in model_names if cv_acc[m] >= args.min_cv]
    if not kept:
        kept = [max(cv_acc, key=cv_acc.get)]
    weights = {m: cv_acc[m] ** args.weight_pow for m in kept}
    wsum = sum(weights.values())
    weights = {m: w / wsum for m, w in weights.items()}
    print(f'\nKept models (cv>={args.min_cv}): {kept}')
    print('Soft-vote weights:', {m: round(w, 3) for m, w in weights.items()})

    # ---- Fit kept models on full augmented training set; predict test with TTA ----
    Xtr_full, ytr_full, _ = augment_grouped(X_tr_raw, y_tr_raw, args.aug_factor, seed=args.seed)
    final_models = make_models(args.seed)
    ens = np.zeros((len(X_test), n_cls))
    test_proba = {}
    member_correct = np.zeros(len(X_test), dtype=int)
    per_model = []
    for m in kept:
        final_models[m].fit(Xtr_full, ytr_full)
        acc = None
        for sh in tta:
            p = final_models[m].predict_proba(np.stack([shift_pad(xi, sh) for xi in X_test]).astype(np.float32))
            acc = p if acc is None else acc + p
        p = acc / len(tta)
        test_proba[m] = p
        ens += weights[m] * p
        member_correct += (p.argmax(1) == y_test).astype(int)
        macc = accuracy_score(y_test, p.argmax(1))
        per_model.append({'model': m, 'cv_acc': round(cv_acc[m], 4), 'weight': round(weights[m], 4),
                          'test_acc': round(float(macc), 4)})

    cv_kept_mean = float(np.mean([cv_acc[m] for m in kept]))

    # ---- Stacking meta-learner: trained on leakage-free OOF probabilities ----
    # Meta-features = concatenated kept-model probabilities. Its own honest score
    # is estimated by group-aware CV on the OOF matrix; then it predicts the test set.
    Z_train = np.concatenate([oof_proba[m] for m in kept], axis=1)
    Z_test = np.concatenate([test_proba[m] for m in kept], axis=1)
    meta = LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced')
    from sklearn.model_selection import cross_val_predict
    meta_oof = cross_val_predict(meta, Z_train, y_tr_raw, cv=args.cv_folds, method='predict')
    stack_cv_acc = accuracy_score(y_tr_raw, meta_oof)
    meta.fit(Z_train, y_tr_raw)
    stack_pred = meta.predict(Z_test)
    stack_test_acc = accuracy_score(y_test, stack_pred)
    print(f'\nStacking meta-learner: CV acc={stack_cv_acc:.4f}  holdout acc={stack_test_acc:.4f}')

    # Pick the final method by honest CV, not by peeking at the holdout.
    if stack_cv_acc > cv_kept_mean + 1e-9:
        y_pred = stack_pred
        final_method = 'stacking'
    else:
        y_pred = ens.argmax(1)
        final_method = 'weighted_vote'
    print(f'Final method selected by CV: {final_method}')

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    vote_test_acc = accuracy_score(y_test, ens.argmax(1))

    print('\n================ HYBRID ENSEMBLE (FINAL) ================')
    print(f'Honest CV: weighted-vote={cv_kept_mean:.4f}  stacking={stack_cv_acc:.4f}  -> using {final_method}')
    print(f'Holdout (weighted-vote)={vote_test_acc:.4f}  (stacking)={stack_test_acc:.4f}')
    print(f'FINAL HOLDOUT test accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)   macro-F1: {test_f1:.4f}')
    print('\nPer-class report (test):')
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print('Confusion matrix:')
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=class_names, columns=class_names).to_string())
    wrong = np.flatnonzero(y_pred != y_test)
    print('\nEnsemble errors:')
    for i in wrong:
        print(f'  test#{i:2d} true={class_names[y_test[i]]:<18} pred={class_names[y_pred[i]]:<18} '
              f'models_correct={member_correct[i]}/{len(kept)} conf={ens[i, y_pred[i]]:.2f}')
    never = np.flatnonzero(member_correct == 0)
    print(f'\nUnrecoverable by kept models: {len(never)} -> {list(never)}  '
          f'(ceiling {(len(y_test)-len(never))/len(y_test):.4f})')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'aug_factor': args.aug_factor, 'cv_folds': args.cv_folds, 'tta': tta,
                   'cv_acc': {m: round(cv_acc[m], 4) for m in model_names},
                   'kept': kept, 'weights': {m: round(weights[m], 4) for m in kept},
                   'cv_kept_mean': round(cv_kept_mean, 4),
                   'stacking_cv_acc': round(float(stack_cv_acc), 4),
                   'stacking_holdout_acc': round(float(stack_test_acc), 4),
                   'weighted_vote_holdout_acc': round(float(vote_test_acc), 4),
                   'final_method': final_method,
                   'holdout_test_acc': round(float(test_acc), 4),
                   'holdout_macro_f1': round(float(test_f1), 4),
                   'per_model': per_model,
                   'never_correct_idx': [int(i) for i in never],
                   'class_names': class_names}, fh, indent=2)
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
