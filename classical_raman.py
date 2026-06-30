"""
classical_raman.py
==================
Classical-ML ensemble for the single-label Raman classifier.

Motivation: with only 123 training spectra, a 1.2M-param CNN is wildly
overparameterised and high-variance (test accuracy swings 73-91% by seed).
A logistic-regression baseline already reaches ~86% on the same holdout.
Classical models with light spectral feature engineering, trained on augmented
data and soft-voted together (with test-time augmentation), are both stronger
and far more stable on data this small.

Holdout is identical to train_cnn_raman.py (train_test_split seed=42, 25% test).

Usage:
    python classical_raman.py
    python classical_raman.py --aug-factor 20 --features sg
"""

import os, json, argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
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


def shift_with_edge_padding(x, shift):
    if shift == 0:
        return x.copy()
    s = np.empty_like(x)
    if shift > 0:
        s[:shift] = x[0]; s[shift:] = x[:-shift]
    else:
        k = -shift; s[-k:] = x[-1]; s[:-k] = x[k:]
    return s


def augment(X, y, factor, seed):
    axis = np.linspace(-1, 1, X.shape[1], dtype=np.float32)
    Xa, ya = [X.copy()], [y.copy()]
    rng = np.random.default_rng(seed)
    for _ in range(factor):
        Xn = np.empty_like(X)
        for i, sp in enumerate(X):
            a = sp.copy()
            a = shift_with_edge_padding(a, int(rng.integers(-SHIFT_MAX, SHIFT_MAX + 1)))
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
        Xa.append(Xn); ya.append(y.copy())
    return np.vstack(Xa), np.concatenate(ya)


def featurize(X, mode):
    """Optional spectral feature engineering. Returns transformed matrix."""
    if mode == 'raw':
        return X
    feats = [X]
    if mode in ('sg', 'all'):
        sg1 = savgol_filter(X, window_length=11, polyorder=3, deriv=1, axis=1)
        feats.append(sg1.astype(np.float32))
    if mode == 'all':
        sg2 = savgol_filter(X, window_length=11, polyorder=3, deriv=2, axis=1)
        feats.append(sg2.astype(np.float32))
    return np.concatenate(feats, axis=1).astype(np.float32)


def load_data():
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


def build_models(seed):
    return {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=5000, C=1.0, class_weight='balanced', solver='lbfgs')),
        'svm_rbf': make_pipeline(StandardScaler(), SVC(
            C=10, gamma='scale', class_weight='balanced', probability=True, random_state=seed)),
        'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance')),
        'rf': RandomForestClassifier(n_estimators=600, random_state=seed,
                                     class_weight='balanced_subsample', n_jobs=-1),
        'histgb': HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=400, random_state=seed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aug-factor', type=int, default=20)
    ap.add_argument('--features', choices=['raw', 'sg', 'all'], default='raw')
    ap.add_argument('--tta', type=str, default='[-3,-2,0,2,3]')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='outputs/logs/classical_results.json')
    args = ap.parse_args()
    tta = [int(float(v)) for v in args.tta.strip('[]').split(',') if v.strip()]

    X, y, class_names = load_data()
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SPLIT_SEED)
    # Train on the full 75% (no inner val needed for classical models)
    X_tr_sp, y_tr = augment(X_train_raw, y_train_raw, args.aug_factor, seed=args.seed)
    X_tr = featurize(X_tr_sp, args.features)

    print(f'Train(aug) {X_tr.shape}  Test {X_test.shape}  features={args.features}  tta={tta}')
    models = build_models(args.seed)

    # Per-model TTA prediction + soft-vote ensemble
    def proba_tta(model, Xsp):
        acc = None
        for sh in tta:
            Xs = np.stack([shift_with_edge_padding(xi, sh) for xi in Xsp]).astype(np.float32)
            p = model.predict_proba(featurize(Xs, args.features))
            acc = p if acc is None else acc + p
        return acc / len(tta)

    rows = []
    ens = np.zeros((len(X_test), len(class_names)))
    member_correct = np.zeros(len(X_test), dtype=int)
    fitted = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        fitted[name] = model
        p = proba_tta(model, X_test)
        ens += p
        acc = accuracy_score(y_test, p.argmax(1))
        member_correct += (p.argmax(1) == y_test).astype(int)
        rows.append({'model': name, 'test_acc': round(float(acc), 4),
                     'macro_f1': round(float(f1_score(y_test, p.argmax(1), average='macro', zero_division=0)), 4)})
        print(f'  {name:10s} test_acc={acc:.4f}')

    ens /= len(models)
    y_pred = ens.argmax(1)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print('\n================ CLASSICAL ENSEMBLE ================')
    print(f'Soft-vote ensemble TEST accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)  macro-F1: {test_f1:.4f}')
    print('\nPer-class report (test):')
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print('Confusion matrix:')
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=class_names, columns=class_names).to_string())
    wrong = np.flatnonzero(y_pred != y_test)
    print('\nEnsemble errors:')
    for i in wrong:
        print(f'  test#{i:2d} true={class_names[y_test[i]]:<18} pred={class_names[y_pred[i]]:<18} '
              f'models_correct={member_correct[i]}/{len(models)} conf={ens[i, y_pred[i]]:.2f}')
    never = np.flatnonzero(member_correct == 0)
    print(f'\nSamples NO model got right (hard ceiling): {len(never)} -> {list(never)}')
    print(f'Max achievable test acc = {(len(y_test)-len(never))/len(y_test):.4f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'features': args.features, 'aug_factor': args.aug_factor, 'tta': tta,
                   'ensemble_test_acc': round(float(test_acc), 4),
                   'ensemble_macro_f1': round(float(test_f1), 4),
                   'per_model': rows, 'never_correct_idx': [int(i) for i in never],
                   'max_achievable_acc': round(float((len(y_test)-len(never))/len(y_test)), 4),
                   'class_names': class_names}, fh, indent=2)
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
