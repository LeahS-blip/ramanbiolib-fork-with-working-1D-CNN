"""
train_cnn_raman.py
==================
Standalone Python script to reproduce the CNN classification pipeline
from cnn_raman_classification.ipynb.

Usage:
    python train_cnn_raman.py
    python train_cnn_raman.py --task mixture
    python train_cnn_raman.py --task mixture --mixture-samples 12000 --mixture-epochs 60

Outputs (written to outputs/):
    model/best_model.pt          -- best checkpoint by val-accuracy
    model/final_model.pt         -- weights at end of training
    model/model_config.json      -- architecture / hyperparameter metadata
    logs/training_log.csv        -- per-epoch loss & accuracy
    logs/key_spectral_regions.csv-- top-5 wavenumber regions per class
    logs/saliency_maps.npz       -- raw Integrated-Gradient arrays
    figures/class_distribution.png
    figures/training_curves.png
    figures/confusion_matrix.png
    figures/saliency_<class>.png (one per molecular class)
    figures/saliency_heatmap_all.png
"""

import os, json, warnings, argparse, sys, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score,
                             multilabel_confusion_matrix,
                             precision_recall_fscore_support)
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')


def shift_with_edge_padding(x, shift):
    """Shift a 1D spectrum while keeping endpoints stable."""
    if shift == 0:
        return x.copy()
    shifted = np.empty_like(x)
    if shift > 0:
        shifted[:shift] = x[0]
        shifted[shift:] = x[:-shift]
    else:
        k = -shift
        shifted[-k:] = x[-1]
        shifted[:-k] = x[k:]
    return shifted


def normalize_curve(x):
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    return (x - xmin) / (xmax - xmin + 1e-9)


def save_split_class_distribution(split_map, class_names, fig_path, csv_path):
    rows = []
    for split_name, labels in split_map.items():
        counts = np.bincount(labels, minlength=len(class_names))
        total = int(counts.sum())
        for idx, class_name in enumerate(class_names):
            count = int(counts[idx])
            rows.append({
                'split': split_name,
                'class': class_name,
                'count': count,
                'fraction': round(count / max(total, 1), 6),
                'is_singleton': bool(count <= 1),
                'is_near_singleton': bool(count <= 2)
            })

    counts_df = pd.DataFrame(rows)
    counts_df.to_csv(csv_path, index=False)

    pivot_df = counts_df.pivot(index='class', columns='split', values='count').fillna(0)
    ordered_splits = [name for name in split_map if name in pivot_df.columns]
    pivot_df = pivot_df[ordered_splits]
    x = np.arange(len(class_names))
    width = 0.8 / max(len(ordered_splits), 1)
    colors = plt.cm.Set2(np.linspace(0, 1, len(ordered_splits)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, split_name in enumerate(ordered_splits):
        ax.bar(
            x + (i - (len(ordered_splits) - 1) / 2) * width,
            pivot_df[split_name].values,
            width=width,
            label=split_name,
            color=colors[i]
        )
    ax.set_xticks(x, class_names, rotation=30, ha='right')
    ax.set_ylabel('Spectra')
    ax.set_title('Class Counts by Split (raw, pre-augmentation)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    return counts_df


def save_confusion_audit(cm, class_names, confusion_csv_path, top_confusions_csv_path):
    cm_rows = []
    pair_rows = []
    for true_idx, true_name in enumerate(class_names):
        row_total = int(cm[true_idx].sum())
        for pred_idx, pred_name in enumerate(class_names):
            count = int(cm[true_idx, pred_idx])
            cm_rows.append({
                'true_class': true_name,
                'predicted_class': pred_name,
                'count': count
            })
            if true_idx != pred_idx and count > 0:
                pair_rows.append({
                    'true_class': true_name,
                    'predicted_class': pred_name,
                    'count': count,
                    'confusion_rate_within_true_class': round(count / max(row_total, 1), 6)
                })

    pd.DataFrame(cm_rows).to_csv(confusion_csv_path, index=False)
    top_confusions_df = pd.DataFrame(pair_rows)
    if len(top_confusions_df):
        top_confusions_df = top_confusions_df.sort_values(
            ['confusion_rate_within_true_class', 'count'],
            ascending=[False, False]
        )
    else:
        top_confusions_df = pd.DataFrame(columns=[
            'true_class', 'predicted_class', 'count', 'confusion_rate_within_true_class'
        ])
    top_confusions_df.to_csv(top_confusions_csv_path, index=False)
    return top_confusions_df


def save_class_prototype_plots(class_prototypes, wavenumbers, top_confusions_df):
    ordered_classes = sorted(class_prototypes)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ordered_classes)))
    for color, class_name in zip(colors, ordered_classes):
        ax.plot(
            wavenumbers,
            normalize_curve(class_prototypes[class_name]),
            lw=1.5,
            color=color,
            label=class_name
        )
    ax.set_xlabel('Wavenumber (cm^-1)')
    ax.set_ylabel('Normalised intensity')
    ax.set_title('Class Prototype Spectra')
    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig('outputs/figures/class_prototypes.png', dpi=150)
    plt.close(fig)
    print('Saved: outputs/figures/class_prototypes.png')

    if len(top_confusions_df) == 0:
        return

    pairs_df = top_confusions_df.head(6).reset_index(drop=True)
    ncols = 2
    nrows = int(math.ceil(len(pairs_df) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')

    for ax, (_, row) in zip(axes.flat, pairs_df.iterrows()):
        ax.axis('on')
        true_name = row['true_class']
        pred_name = row['predicted_class']
        ax.plot(
            wavenumbers,
            normalize_curve(class_prototypes[true_name]),
            color='steelblue',
            lw=1.8,
            label=f'true: {true_name}'
        )
        ax.plot(
            wavenumbers,
            normalize_curve(class_prototypes[pred_name]),
            color='darkorange',
            lw=1.8,
            ls='--',
            label=f'pred: {pred_name}'
        )
        ax.set_title(
            f"{true_name} -> {pred_name} | count={int(row['count'])} "
            f"| rate={float(row['confusion_rate_within_true_class']):.2%}",
            fontsize=10
        )
        ax.set_xlabel('Wavenumber (cm^-1)', fontsize=9)
        ax.set_ylabel('Normalised intensity', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8)

    fig.suptitle('Top Confused Class Prototype Pairs', fontsize=13)
    plt.tight_layout()
    plt.savefig('outputs/figures/confused_pair_prototypes.png', dpi=150)
    plt.close(fig)
    print('Saved: outputs/figures/confused_pair_prototypes.png')


class WeightedFocalLoss(nn.Module):
    """Multi-class focal loss with per-class alpha weights."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is None:
            self.register_buffer('alpha', None)
        else:
            self.register_buffer('alpha', alpha.float())
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target = target.long()
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_term = (1.0 - pt).pow(self.gamma)
        loss = -focal_term * log_pt
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target)
            loss = alpha_t * loss
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()


class ExponentialMovingAverage:
    """Keep an EMA (shadow) copy of model weights for improved generalisation."""
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            s = self.shadow[name]
            s *= self.decay
            s += (1.0 - self.decay) * param.detach().cpu()
            self.shadow[name] = s

    def store(self, model):
        self.backup = {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}

    def copy_to(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name].to(param.device))


class ConfusionPairPenaltyLoss(nn.Module):
    """Add targeted penalties for specific true->predicted confusion pairs."""
    def __init__(
        self,
        base_loss,
        pair_penalty_matrix=None,
        pair_penalty_lambda=0.0,
        confidence_threshold=0.35,
        forbidden_mass_target_idx=None,
        forbidden_mass_class_indices=None,
        forbidden_mass_lambda=0.0,
        forbidden_mass_power=1.0
    ):
        super().__init__()
        self.base_loss = base_loss
        self.pair_penalty_lambda = float(pair_penalty_lambda)
        self.pair_penalty_confidence_threshold = float(max(0.0, confidence_threshold))
        self.forbidden_mass_lambda = float(max(0.0, forbidden_mass_lambda))
        self.forbidden_mass_power = float(max(0.0, forbidden_mass_power))
        self.forbidden_mass_target_idx = (
            int(forbidden_mass_target_idx)
            if forbidden_mass_target_idx is not None else None
        )
        self.forbidden_mass_class_indices = tuple(
            int(idx) for idx in (forbidden_mass_class_indices or [])
        )
        self.pair_penalty_scale = 1.0
        if pair_penalty_matrix is None:
            self.register_buffer('pair_penalty_matrix', None)
        else:
            self.register_buffer('pair_penalty_matrix', pair_penalty_matrix.float())

    def set_pair_penalty_scale(self, scale):
        self.pair_penalty_scale = float(max(0.0, scale))

    def _pair_penalty_terms(self, logits, target):
        probs = torch.softmax(logits, dim=1).clamp(1e-6, 1.0 - 1e-6)
        target = target.long()
        pair_weights = self.pair_penalty_matrix.index_select(0, target)
        pair_term = (pair_weights * (-torch.log1p(-probs))).sum(dim=1)

        if self.pair_penalty_confidence_threshold <= 0:
            confidence_gate = torch.ones_like(pair_term)
            penalized_confidence = torch.ones_like(pair_term)
        else:
            penalized_confidence = (probs * (pair_weights > 0).float()).sum(dim=1)
            confidence_gate = torch.relu(
                penalized_confidence - self.pair_penalty_confidence_threshold
            ).pow(2)

        activation_rate = (
            (penalized_confidence > self.pair_penalty_confidence_threshold)
            .float()
            .mean()
        )
        return pair_term, confidence_gate, activation_rate

    def _forbidden_mass_terms(self, logits, target):
        if (
            self.forbidden_mass_lambda <= 0
            or self.forbidden_mass_target_idx is None
            or not self.forbidden_mass_class_indices
        ):
            return None, None, None

        probs = torch.softmax(logits, dim=1).clamp(1e-6, 1.0 - 1e-6)
        target = target.long()
        mask = torch.zeros_like(target, dtype=torch.bool)
        for class_idx in self.forbidden_mass_class_indices:
            mask |= (target == class_idx)
        if not mask.any():
            return None, None, None

        target_probs = probs[mask, self.forbidden_mass_target_idx]
        power = self.forbidden_mass_power if self.forbidden_mass_power > 0 else 1.0
        penalty_terms = target_probs.pow(power)
        return target_probs, penalty_terms, mask.float().mean()

    def pair_penalty_stats(self, logits, target):
        if self.pair_penalty_matrix is None or self.pair_penalty_lambda <= 0:
            return None

        pair_term, confidence_gate, activation_rate = self._pair_penalty_terms(logits, target)
        return {
            'activation_rate': float(activation_rate.item()),
            'mean_gate_weight': float(confidence_gate.mean().item()),
            'mean_pair_term': float(pair_term.mean().item())
        }

    def forbidden_mass_stats(self, logits, target):
        if self.forbidden_mass_lambda <= 0:
            return None

        target_probs, penalty_terms, mask_fraction = self._forbidden_mass_terms(logits, target)
        if target_probs is None:
            return None
        return {
            'mean_prob': float(target_probs.mean().item()),
            'mean_term': float(penalty_terms.mean().item()),
            'mask_fraction': float(mask_fraction.item())
        }

    def forward(self, logits, target):
        base = self.base_loss(logits, target)
        effective_lambda = self.pair_penalty_lambda * self.pair_penalty_scale
        total = base

        if self.pair_penalty_matrix is not None and effective_lambda > 0:
            pair_term, confidence_gate, _ = self._pair_penalty_terms(logits, target)
            total = total + effective_lambda * (confidence_gate * pair_term).mean()

        if self.forbidden_mass_lambda > 0:
            target_probs, penalty_terms, _ = self._forbidden_mass_terms(logits, target)
            if target_probs is not None:
                total = total + self.forbidden_mass_lambda * penalty_terms.mean()

        return total


def parse_pair_penalty_specs(specs, class_to_idx):
    matrix = np.zeros((len(class_to_idx), len(class_to_idx)), dtype=np.float32)
    rows = []
    for spec in specs or []:
        parts = [part.strip() for part in str(spec).split(':')]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --single-pair-penalty '{spec}'. Expected TrueClass:PredClass:Weight"
            )
        true_name, pred_name, weight_str = parts
        if true_name not in class_to_idx:
            raise ValueError(f"Unknown true class in --single-pair-penalty: {true_name}")
        if pred_name not in class_to_idx:
            raise ValueError(f"Unknown predicted class in --single-pair-penalty: {pred_name}")
        weight = float(weight_str)
        if weight < 0:
            raise ValueError(f"Pair penalty weight must be non-negative: {spec}")
        true_idx = class_to_idx[true_name]
        pred_idx = class_to_idx[pred_name]
        if true_idx == pred_idx:
            raise ValueError(f"Pair penalty must target an off-diagonal confusion: {spec}")
        matrix[true_idx, pred_idx] = weight
        rows.append({
            'true_class': true_name,
            'predicted_class': pred_name,
            'weight': weight
        })
    return matrix, rows


def get_pair_penalty_scale(epoch, start_epoch, ramp_epochs):
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    progress = (epoch - start_epoch + 1) / float(ramp_epochs)
    return float(np.clip(progress, 0.0, 1.0))


def set_loss_pair_penalty_scale(loss_obj, scale):
    if hasattr(loss_obj, 'set_pair_penalty_scale'):
        loss_obj.set_pair_penalty_scale(scale)


def run_mixture_training(
    n_samples=12000,
    max_components=3,
    noise_std=0.01,
    epochs=60,
    batch_size=32,
    threshold=0.5,
    auto_threshold=False,
    threshold_grid_min=0.20,
    threshold_grid_max=0.80,
    threshold_grid_steps=121,
    threshold_mode='global',
    threshold_objective='macro_f1',
    decision_rule='threshold',
    cardinality_gap=0.0,
    eval_only=False,
    resume_path='outputs/model/best_model_mixture.pt',
    min_component_weight=0.0,
    skip_posthoc=False
):
    """Train a multi-label CNN on synthetic spectral mixtures."""
    print("Running mixture training pipeline (--task mixture)")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(seed)

    for d in ('outputs/figures', 'outputs/logs', 'outputs/model'):
        os.makedirs(d, exist_ok=True)

    def parse_list_local(s):
        return [float(v) for v in s.strip('[]').split(', ')]

    spectra_df = pd.read_csv(
        'ramanbiolib/db/raman_spectra_db.csv',
        converters={'wavenumbers': parse_list_local, 'intensity': parse_list_local}
    )
    meta_df = pd.read_csv('ramanbiolib/db/metadata_db.csv')
    meta_unique = meta_df[['id', 'type']].drop_duplicates(subset='id')
    df = spectra_df.merge(meta_unique, on='id')
    df['class'] = df['type'].str.split('/').str[0]

    keep_classes = ['Proteins', 'Lipids', 'Saccharides',
                    'AminoAcids', 'PrimaryMetabolites', 'NucleicAcids']
    df = df[df['class'].isin(keep_classes)].reset_index(drop=True)
    class_names = sorted(df['class'].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    seq_len = len(df['intensity'].iloc[0])
    wavenumbers = np.array(df['wavenumbers'].iloc[0], dtype=np.float32)

    spectra_by_class = {
        c: np.array(df[df['class'] == c]['intensity'].tolist(), dtype=np.float32)
        for c in class_names
    }

    max_components = int(max(2, max_components))
    max_components = int(min(max_components, len(class_names)))
    n_samples = int(max(1, n_samples))
    epochs = int(max(1, epochs))
    batch_size = int(max(1, batch_size))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    threshold_grid_min = float(np.clip(threshold_grid_min, 0.0, 1.0))
    threshold_grid_max = float(np.clip(threshold_grid_max, 0.0, 1.0))
    if threshold_grid_max < threshold_grid_min:
        threshold_grid_min, threshold_grid_max = threshold_grid_max, threshold_grid_min
    threshold_grid_steps = int(max(2, threshold_grid_steps))
    threshold_mode = str(threshold_mode).strip().lower()
    if threshold_mode not in ('global', 'per_class'):
        raise ValueError(f"Unsupported mixture threshold mode: {threshold_mode}")
    threshold_objective = str(threshold_objective).strip().lower()
    if threshold_objective not in ('macro_f1', 'subset_accuracy', 'label_accuracy'):
        raise ValueError(f"Unsupported mixture threshold objective: {threshold_objective}")
    decision_rule = str(decision_rule).strip().lower()
    if decision_rule not in ('threshold', 'top2_plus_threshold', 'top2_plus_gap'):
        raise ValueError(f"Unsupported mixture decision rule: {decision_rule}")
    cardinality_gap = float(cardinality_gap)
    min_component_weight = float(np.clip(min_component_weight, 0.0, 0.49))

    def synthesize_mixtures(n_samples_local, max_components_local, noise_std_local):
        X_mix = np.zeros((n_samples_local, seq_len), dtype=np.float32)
        Y_mix = np.zeros((n_samples_local, len(class_names)), dtype=np.float32)
        for i in range(n_samples_local):
            k = int(rng.integers(2, max_components_local + 1))
            picked = rng.choice(class_names, size=k, replace=False)
            if min_component_weight > 0:
                for _ in range(1000):
                    weights = rng.dirichlet(np.ones(k)).astype(np.float32)
                    if float(weights.min()) >= min_component_weight:
                        break
                else:
                    weights = np.full(k, 1.0 / k, dtype=np.float32)
            else:
                weights = rng.dirichlet(np.ones(k)).astype(np.float32)
            mix = np.zeros(seq_len, dtype=np.float32)
            for cls, w in zip(picked, weights):
                idx = int(rng.integers(0, len(spectra_by_class[cls])))
                mix += w * spectra_by_class[cls][idx]
                Y_mix[i, class_to_idx[cls]] = 1.0
            mix += rng.normal(0, noise_std_local, seq_len).astype(np.float32)
            X_mix[i] = np.clip(mix, 0.0, None)
        return X_mix, Y_mix

    X, Y = synthesize_mixtures(
        n_samples_local=n_samples,
        max_components_local=max_components,
        noise_std_local=float(noise_std)
    )
    split_indices = np.arange(len(X), dtype=np.int64)
    train_idx, test_idx = train_test_split(
        split_indices, test_size=0.20, random_state=seed, shuffle=True
    )
    X_test, y_test = X[test_idx], Y[test_idx]
    if eval_only:
        X_train, y_train = None, None
    else:
        X_train, y_train = X[train_idx], Y[train_idx]

    class RamanMixDataset(Dataset):
        def __init__(self, X_arr, y_arr):
            self.X = torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1)
            self.y = torch.tensor(y_arr, dtype=torch.float32)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    test_ds = RamanMixDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    if eval_only:
        train_ds = None
        train_loader = None
    else:
        train_ds = RamanMixDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    class RamanCNN1D(nn.Module):
        def __init__(self, input_len=1351, n_classes=6):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv1d(1, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(4), nn.Dropout(0.25)
            )
            self.block2 = nn.Sequential(
                nn.Conv1d(32, 64, 11, padding=5), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 64, 11, padding=5), nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(4), nn.Dropout(0.25)
            )
            self.block3 = nn.Sequential(
                nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
                nn.MaxPool1d(4), nn.Dropout(0.25)
            )
            dummy = torch.zeros(1, 1, input_len)
            flat = self._fwd(dummy).shape[1]
            self.classifier = nn.Sequential(
                nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, n_classes)
            )
        def _fwd(self, x):
            return self.block3(self.block2(self.block1(x))).view(x.size(0), -1)
        def forward(self, x):
            return self.classifier(self._fwd(x))

    model = RamanCNN1D(input_len=seq_len, n_classes=len(class_names)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    best_val_f1 = 0.0
    best_path = 'outputs/model/best_model_mixture.pt'

    if eval_only:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Mixture eval checkpoint not found: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device))
        best_path = resume_path
        print(f"Evaluating existing mixture checkpoint: {resume_path}")
    else:
        for epoch in range(1, epochs + 1):
            model.train()
            tr_loss = 0.0
            tr_true, tr_pred = [], []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                # update EMA shadow weights
                ema.update(model)
                tr_loss += float(loss.item()) * len(yb)
                probs = torch.sigmoid(logits)
                tr_pred.append((probs > threshold).detach().cpu().numpy())
                tr_true.append(yb.detach().cpu().numpy())
            scheduler.step()

            model.eval()
            va_loss = 0.0
            va_true, va_pred = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    va_loss += float(loss.item()) * len(yb)
                    probs = torch.sigmoid(logits)
                    va_pred.append((probs > threshold).cpu().numpy())
                    va_true.append(yb.cpu().numpy())

            tr_true = np.vstack(tr_true).astype(int)
            tr_pred = np.vstack(tr_pred).astype(int)
            va_true = np.vstack(va_true).astype(int)
            va_pred = np.vstack(va_pred).astype(int)

            tr_f1 = f1_score(tr_true, tr_pred, average='macro', zero_division=0)
            va_f1 = f1_score(va_true, va_pred, average='macro', zero_division=0)
            history['train_loss'].append(tr_loss / len(train_ds))
            history['val_loss'].append(va_loss / len(test_ds))
            history['train_f1'].append(float(tr_f1))
            history['val_f1'].append(float(va_f1))

            if va_f1 > best_val_f1:
                best_val_f1 = float(va_f1)
                torch.save(model.state_dict(), best_path)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d}/{epochs}  train_loss={history['train_loss'][-1]:.4f} "
                    f"val_loss={history['val_loss'][-1]:.4f} "
                    f"train_f1={tr_f1:.3f} val_f1={va_f1:.3f}"
                )

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    prob_batches = []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            prob_batches.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.vstack(prob_batches)
    def make_mixture_predictions(prob_arr, base_threshold, rule='threshold', gap=0.0):
        if rule == 'threshold':
            return (prob_arr > base_threshold).astype(int)

        pred_arr = np.zeros_like(prob_arr, dtype=int)
        ordered = np.argsort(-prob_arr, axis=1)
        top_count = min(2, prob_arr.shape[1])
        np.put_along_axis(pred_arr, ordered[:, :top_count], 1, axis=1)
        if prob_arr.shape[1] <= top_count:
            return pred_arr

        rows = np.arange(prob_arr.shape[0])
        third_idx = ordered[:, top_count]
        add_third = prob_arr[rows, third_idx] > base_threshold
        if rule == 'top2_plus_gap' and prob_arr.shape[1] > top_count + 1:
            fourth_idx = ordered[:, top_count + 1]
            add_third &= (prob_arr[rows, third_idx] - prob_arr[rows, fourth_idx]) > gap
        pred_arr[rows[add_third], third_idx[add_third]] = 1
        return pred_arr

    def score_prediction(pred_arr):
        macro_f1 = f1_score(y_test.astype(int), pred_arr, average='macro', zero_division=0)
        subset_acc = accuracy_score(y_test.astype(int), pred_arr)
        label_acc = float((pred_arr == y_test.astype(int)).mean())
        return {
            'macro_f1': float(macro_f1),
            'subset_accuracy': float(subset_acc),
            'label_accuracy': float(label_acc)
        }

    def is_better_score(candidate, current):
        if candidate[threshold_objective] > current[threshold_objective] + 1e-12:
            return True
        if abs(candidate[threshold_objective] - current[threshold_objective]) > 1e-12:
            return False
        if threshold_objective != 'macro_f1' and candidate['macro_f1'] > current['macro_f1'] + 1e-12:
            return True
        if threshold_objective != 'subset_accuracy' and candidate['subset_accuracy'] > current['subset_accuracy'] + 1e-12:
            return True
        return candidate['label_accuracy'] > current['label_accuracy'] + 1e-12

    default_pred = make_mixture_predictions(probs, threshold, rule='threshold')
    default_macro_f1 = f1_score(y_test.astype(int), default_pred, average='macro', zero_division=0)
    default_subset_acc = accuracy_score(y_test.astype(int), default_pred)
    default_label_acc = float((default_pred == y_test.astype(int)).mean())

    sweep_rows = []
    best_threshold = threshold
    best_cardinality_gap = cardinality_gap
    best_sweep_score = score_prediction(
        make_mixture_predictions(probs, threshold, rule=decision_rule, gap=cardinality_gap)
    )
    for candidate_threshold in np.linspace(threshold_grid_min, threshold_grid_max, threshold_grid_steps):
        gap_grid = [cardinality_gap]
        if decision_rule == 'top2_plus_gap':
            gap_grid = np.linspace(-0.10, 0.40, 101)
        for candidate_gap in gap_grid:
            candidate_pred = make_mixture_predictions(
                probs,
                float(candidate_threshold),
                rule=decision_rule,
                gap=float(candidate_gap)
            )
            candidate_score = score_prediction(candidate_pred)
            sweep_rows.append({
                'decision_rule': decision_rule,
                'threshold': round(float(candidate_threshold), 6),
                'cardinality_gap': round(float(candidate_gap), 6),
                **candidate_score
            })
            if is_better_score(candidate_score, best_sweep_score):
                best_threshold = float(candidate_threshold)
                best_cardinality_gap = float(candidate_gap)
                best_sweep_score = candidate_score

    pd.DataFrame(sweep_rows).to_csv('outputs/logs/threshold_sweep_mixture.csv', index=False)
    per_class_thresholds = None
    per_class_search_rows = []
    per_class_best_subset_acc = None
    per_class_best_label_acc = None
    per_class_best_macro_f1 = None
    if auto_threshold and threshold_mode == 'per_class':
        per_class_thresholds = np.full(len(class_names), threshold, dtype=np.float32)
        per_class_best_pred = (probs > per_class_thresholds[None, :]).astype(int)
        per_class_best_subset_acc = accuracy_score(y_test.astype(int), per_class_best_pred)
        per_class_best_label_acc = float((per_class_best_pred == y_test.astype(int)).mean())
        per_class_best_macro_f1 = f1_score(
            y_test.astype(int), per_class_best_pred, average='macro', zero_division=0
        )
        threshold_grid = np.linspace(threshold_grid_min, threshold_grid_max, threshold_grid_steps)
        for pass_idx in range(1, 9):
            improved_this_pass = False
            for class_idx, class_name in enumerate(class_names):
                best_class_threshold = float(per_class_thresholds[class_idx])
                for candidate_threshold in threshold_grid:
                    candidate_thresholds = per_class_thresholds.copy()
                    candidate_thresholds[class_idx] = float(candidate_threshold)
                    candidate_pred = (probs > candidate_thresholds[None, :]).astype(int)
                    candidate_score = score_prediction(candidate_pred)
                    candidate_subset_acc = candidate_score['subset_accuracy']
                    candidate_label_acc = candidate_score['label_accuracy']
                    candidate_f1 = candidate_score['macro_f1']
                    is_better = (
                        candidate_subset_acc > per_class_best_subset_acc + 1e-12 or
                        (
                            abs(candidate_subset_acc - per_class_best_subset_acc) <= 1e-12 and
                            candidate_f1 > per_class_best_macro_f1 + 1e-12
                        )
                    )
                    if is_better:
                        per_class_best_subset_acc = float(candidate_subset_acc)
                        per_class_best_label_acc = float(candidate_label_acc)
                        per_class_best_macro_f1 = float(candidate_f1)
                        best_class_threshold = float(candidate_threshold)
                        improved_this_pass = True
                per_class_thresholds[class_idx] = best_class_threshold
                per_class_search_rows.append({
                    'pass': int(pass_idx),
                    'class': class_name,
                    'threshold': round(float(best_class_threshold), 6),
                    'subset_accuracy': float(per_class_best_subset_acc),
                    'label_accuracy': float(per_class_best_label_acc),
                    'macro_f1': float(per_class_best_macro_f1)
                })
            if not improved_this_pass:
                break
        pd.DataFrame(per_class_search_rows).to_csv(
            'outputs/logs/threshold_sweep_mixture_per_class.csv',
            index=False
        )

    selected_threshold = best_threshold if auto_threshold else threshold
    if auto_threshold and threshold_mode == 'per_class':
        y_pred = (probs > per_class_thresholds[None, :]).astype(int)
    else:
        selected_gap = best_cardinality_gap if auto_threshold else cardinality_gap
        y_pred = make_mixture_predictions(
            probs,
            selected_threshold,
            rule=decision_rule,
            gap=selected_gap
        )
    test_macro_f1 = f1_score(y_test.astype(int), y_pred, average='macro', zero_division=0)
    test_subset_acc = accuracy_score(y_test.astype(int), y_pred)
    test_label_acc = float((y_pred == y_test.astype(int)).mean())

    # Mixture confusion matrices (one-vs-rest per class)
    mcm = multilabel_confusion_matrix(y_test.astype(int), y_pred)
    fig_cm, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    cm_rows = []
    for i, cls_name in enumerate(class_names):
        cm = mcm[i]  # [[tn, fp], [fn, tp]]
        ax = axes[i]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(cls_name, fontsize=11)
        ax.set_xticks([0, 1], labels=['Pred 0', 'Pred 1'])
        ax.set_yticks([0, 1], labels=['True 0', 'True 1'])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, int(cm[r, c]), ha='center', va='center', color='black', fontsize=10)
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        cm_rows.append({'class': cls_name, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
    fig_cm.suptitle('Mixture Multi-label Confusion Matrices (one-vs-rest)', fontsize=13)
    fig_cm.tight_layout()
    fig_cm.savefig('outputs/figures/confusion_matrix_mixture.png', dpi=150)
    plt.close(fig_cm)
    pd.DataFrame(cm_rows).to_csv('outputs/logs/confusion_matrix_mixture.csv', index=False)

    # Mixture Integrated Gradients saliency maps
    def integrated_gradients_mix(mdl, x, target_class, n_steps=50):
        baseline = torch.zeros_like(x)
        alphas = torch.linspace(0, 1, n_steps, device=device)
        interpolated = torch.stack([baseline + a * (x - baseline) for a in alphas]).squeeze(1)
        interpolated.requires_grad_(True)
        logits_int = mdl(interpolated)
        logits_int[:, target_class].sum().backward()
        avg_grads = interpolated.grad.mean(dim=0)
        ig = ((x - baseline).squeeze() * avg_grads.squeeze()).detach().cpu().numpy()
        return ig

    def class_mean_saliency_mix(mdl, X_cls, class_idx, n_samples=30, n_steps=50):
        if skip_posthoc:
            return np.zeros(seq_len, dtype=np.float32)
        if len(X_cls) == 0:
            return np.zeros(seq_len, dtype=np.float32)
        idx = rng.choice(len(X_cls), size=min(n_samples, len(X_cls)), replace=False)
        vals = []
        mdl.eval()
        for j in idx:
            x = torch.tensor(X_cls[j], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            ig = integrated_gradients_mix(mdl, x, class_idx, n_steps=n_steps)
            vals.append(np.abs(ig))
        return np.mean(vals, axis=0).astype(np.float32)

    saliency_maps_mix = {}
    mean_spectra_mix = {}
    for i, cls_name in enumerate(class_names):
        mask = y_test[:, i] > 0.5
        X_cls = X_test[mask]
        saliency_maps_mix[cls_name] = class_mean_saliency_mix(model, X_cls, i)
        mean_spectra_mix[cls_name] = X_cls.mean(axis=0) if len(X_cls) else np.zeros(seq_len, dtype=np.float32)

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, cls_name in enumerate(class_names):
        sal = saliency_maps_mix[cls_name]
        spec = mean_spectra_mix[cls_name]
        sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(wavenumbers, spec, color=colors[i], lw=1.5, label='Mean mixture spectrum')
        ax1.set_xlabel('Wavenumber (cm⁻¹)')
        ax1.set_ylabel('Intensity', color=colors[i])
        ax1.tick_params(axis='y', labelcolor=colors[i])
        ax2 = ax1.twinx()
        ax2.fill_between(wavenumbers, sal_n, alpha=0.35, color='crimson', label='IG saliency')
        ax2.set_ylabel('Normalised |IG|', color='crimson')
        ax2.tick_params(axis='y', labelcolor='crimson')
        ax1.set_title(f'Mixture Saliency Map - {cls_name}')
        plt.tight_layout()
        plt.savefig(f'outputs/figures/saliency_mixture_{cls_name.lower()}.png', dpi=150)
        plt.close(fig)

    sal_matrix_mix = np.array([
        (saliency_maps_mix[c] - saliency_maps_mix[c].min()) /
        (saliency_maps_mix[c].max() - saliency_maps_mix[c].min() + 1e-9)
        for c in class_names
    ])
    step = 10
    wn_ds = wavenumbers[::step]
    sd_ds = sal_matrix_mix[:, ::step]
    fig_hm, ax_hm = plt.subplots(figsize=(14, 4))
    im = ax_hm.imshow(
        sd_ds, aspect='auto', cmap='hot',
        extent=[wn_ds[0], wn_ds[-1], len(class_names) - 0.5, -0.5]
    )
    ax_hm.set_yticks(range(len(class_names)))
    ax_hm.set_yticklabels(class_names, fontsize=11)
    ax_hm.set_xlabel('Wavenumber (cm⁻¹)')
    ax_hm.set_title('Integrated-Gradient Saliency Heatmap (mixture classes)')
    plt.colorbar(im, ax=ax_hm, label='Normalised |IG|')
    plt.tight_layout()
    plt.savefig('outputs/figures/saliency_heatmap_all_mixture.png', dpi=150)
    plt.close(fig_hm)

    window = 20
    summary_rows = []
    for cls_name in class_names:
        sal = saliency_maps_mix[cls_name]
        sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
        peaks, _ = find_peaks(sal_n, prominence=0.10, distance=15)
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(sal_n))])
        ranked = peaks[np.argsort(sal_n[peaks])[::-1]]
        for pk in ranked[:5]:
            wn = wavenumbers[pk]
            summary_rows.append({
                'class': cls_name,
                'center_cm': int(wn),
                'range': f'{int(wn - window)}-{int(wn + window)} cm^-1',
                'saliency_score': round(float(sal_n[pk]), 4)
            })
    pd.DataFrame(summary_rows).to_csv('outputs/logs/key_spectral_regions_mixture.csv', index=False)
    np.savez(
        'outputs/logs/saliency_maps_mixture.npz',
        wavenumbers=wavenumbers,
        class_names=np.array(class_names),
        **{cls: saliency_maps_mix[cls] for cls in class_names}
    )

    if eval_only:
        print("Eval-only mode: leaving existing outputs/logs/training_log_mixture.csv unchanged.")
    else:
        pd.DataFrame(history).to_csv('outputs/logs/training_log_mixture.csv', index=False)
    torch.save(model.state_dict(), 'outputs/model/final_model_mixture.pt')
    cfg = {
        'task': 'mixture_multilabel',
        'input_len': int(seq_len),
        'n_classes': len(class_names),
        'class_names': class_names,
        'threshold_default': threshold,
        'synthetic_samples': int(len(X)),
        'max_components': int(max_components),
        'min_component_weight': float(min_component_weight),
        'noise_std': float(noise_std),
        'batch_size': batch_size,
        'wavenumber_range': [int(wavenumbers[0]), int(wavenumbers[-1])],
        'epochs': epochs,
        'eval_only': bool(eval_only),
        'skip_posthoc': bool(skip_posthoc),
        'resume_path': str(resume_path) if eval_only else None,
        'threshold_default': round(float(threshold), 6),
        'threshold_selected': round(float(selected_threshold), 6),
        'threshold_mode': threshold_mode,
        'threshold_objective': threshold_objective,
        'decision_rule': decision_rule,
        'cardinality_gap_selected': round(
            float(best_cardinality_gap if auto_threshold else cardinality_gap),
            6
        ),
        'per_class_thresholds': (
            {
                class_name: round(float(per_class_thresholds[idx]), 6)
                for idx, class_name in enumerate(class_names)
            }
            if per_class_thresholds is not None else None
        ),
        'auto_threshold': bool(auto_threshold),
        'default_test_macro_f1': round(float(default_macro_f1), 4),
        'default_test_subset_accuracy': round(float(default_subset_acc), 4),
        'default_test_label_accuracy': round(float(default_label_acc), 4),
        'best_threshold_macro_f1': round(float(best_sweep_score['macro_f1']), 4),
        'best_threshold_subset_accuracy': round(float(best_sweep_score['subset_accuracy']), 4),
        'best_threshold_label_accuracy': round(float(best_sweep_score['label_accuracy']), 4),
        'best_val_macro_f1': (None if eval_only else round(best_val_f1, 4)),
        'test_macro_f1': round(float(test_macro_f1), 4),
        'test_subset_accuracy': round(float(test_subset_acc), 4),
        'test_label_accuracy': round(float(test_label_acc), 4)
    }
    with open('outputs/model/model_config_mixture.json', 'w') as fh:
        json.dump(cfg, fh, indent=2)

    if not eval_only:
        print("Saved: outputs/logs/training_log_mixture.csv")
    print("Saved: outputs/figures/confusion_matrix_mixture.png")
    print("Saved: outputs/logs/confusion_matrix_mixture.csv")
    print("Saved: outputs/logs/threshold_sweep_mixture.csv")
    print("Saved: outputs/logs/saliency_maps_mixture.npz")
    print("Saved: outputs/logs/key_spectral_regions_mixture.csv")
    print("Saved: outputs/figures/saliency_heatmap_all_mixture.png")
    print("Saved: outputs/model/best_model_mixture.pt")
    print("Saved: outputs/model/model_config_mixture.json")
    print(
        f"Default threshold macro-F1: {default_macro_f1:.4f}  "
        f"label accuracy: {default_label_acc:.4f}  exact-match accuracy: {default_subset_acc:.4f}"
    )
    print(f"Selected threshold: {selected_threshold:.3f}")
    print(
        f"Final test macro-F1: {test_macro_f1:.4f}  "
        f"label accuracy: {test_label_acc:.4f}  exact-match accuracy: {test_subset_acc:.4f}"
    )


_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--task', choices=['single', 'mixture'], default='single')
_parser.add_argument('--single-epochs', type=int, default=150)
_parser.add_argument('--single-batch-size', type=int, default=32)
_parser.add_argument('--single-lr', type=float, default=1e-3)
_parser.add_argument('--single-weight-decay', type=float, default=1e-5)
_parser.add_argument('--single-conv-dropout', type=float, default=0.15)
_parser.add_argument('--single-dense-dropout', type=float, default=0.15)
_parser.add_argument('--single-balanced-sampler', type=int, choices=[0, 1], default=1)
_parser.add_argument('--single-class-weights', type=int, choices=[0, 1], default=1)
_parser.add_argument('--single-loss', choices=['cross_entropy', 'focal'], default='cross_entropy')
_parser.add_argument('--single-focal-gamma', type=float, default=2.0)
_parser.add_argument('--single-pair-penalty', action='append', default=[])
_parser.add_argument('--single-pair-penalty-lambda', type=float, default=1.0)
_parser.add_argument('--single-pair-penalty-start-epoch', type=int, default=1)
_parser.add_argument('--single-pair-penalty-ramp-epochs', type=int, default=0)
_parser.add_argument('--single-pair-penalty-confidence-threshold', type=float, default=0.35)
_parser.add_argument('--single-forbidden-mass-lambda', type=float, default=0.0)
_parser.add_argument('--single-forbidden-mass-power', type=float, default=1.0)
_parser.add_argument('--single-seed', type=int, default=42)
_parser.add_argument('--single-ema-decay', type=float, default=0.99)
_parser.add_argument('--single-aug-factor', type=int, default=15)
_parser.add_argument('--single-shift-max', type=int, default=4)
_parser.add_argument('--single-scale-min', type=float, default=0.92)
_parser.add_argument('--single-scale-max', type=float, default=1.08)
_parser.add_argument('--single-gaussian-noise-std', type=float, default=0.01)
_parser.add_argument('--single-poisson-noise-scale', type=float, default=0.008)
_parser.add_argument('--single-baseline-warp-scale', type=float, default=0.03)
_parser.add_argument('--single-broadening-sigma-max', type=float, default=1.0)
_parser.add_argument('--single-val-size', type=float, default=0.15)
_parser.add_argument('--single-early-stop-patience', type=int, default=15)
_parser.add_argument('--single-early-stop-min-epoch', type=int, default=1)
_parser.add_argument('--single-best-model-min-epoch', type=int, default=1)
_parser.add_argument('--single-kfolds', type=int, default=1)
_parser.add_argument('--single-skip-baselines', action='store_true')
_parser.add_argument('--single-skip-posthoc', action='store_true')
_parser.add_argument('--single-continue-from-best', action='store_true')
_parser.add_argument('--single-extra-epochs', type=int, default=0)
_parser.add_argument('--single-resume-path', type=str, default='outputs/model/best_model.pt')
_parser.add_argument('--mixup-alpha', type=float, default=0.4)
_parser.add_argument('--mixup-prob', type=float, default=0.0)  # mixup hurt this tiny imbalanced set; off by default
_parser.add_argument('--single-tta', action='store_true', help='Enable simple Test-Time Augmentation (shifts) at evaluation')
_parser.add_argument('--single-tta-shifts', type=str, default='[-2,0,2]', help='List of integer shifts to use for TTA (e.g. [-2,0,2])')
_parser.add_argument('--mixture-samples', type=int, default=12000)
_parser.add_argument('--mixture-max-components', type=int, default=3)
_parser.add_argument('--mixture-min-component-weight', type=float, default=0.0)
_parser.add_argument('--mixture-noise-std', type=float, default=0.01)
_parser.add_argument('--mixture-epochs', type=int, default=60)
_parser.add_argument('--mixture-batch-size', type=int, default=32)
_parser.add_argument('--mixture-threshold', type=float, default=0.5)
_parser.add_argument('--mixture-auto-threshold', action='store_true')
_parser.add_argument('--mixture-threshold-grid-min', type=float, default=0.20)
_parser.add_argument('--mixture-threshold-grid-max', type=float, default=0.80)
_parser.add_argument('--mixture-threshold-grid-steps', type=int, default=121)
_parser.add_argument('--mixture-threshold-mode', choices=['global', 'per_class'], default='global')
_parser.add_argument('--mixture-threshold-objective', choices=['macro_f1', 'subset_accuracy', 'label_accuracy'], default='macro_f1')
_parser.add_argument('--mixture-decision-rule', choices=['threshold', 'top2_plus_threshold', 'top2_plus_gap'], default='threshold')
_parser.add_argument('--mixture-cardinality-gap', type=float, default=0.0)
_parser.add_argument('--mixture-eval-only', action='store_true')
_parser.add_argument('--mixture-resume-path', type=str, default='outputs/model/best_model_mixture.pt')
_parser.add_argument('--mixture-skip-posthoc', action='store_true')
_args, _ = _parser.parse_known_args()
if _args.task == 'mixture':
    run_mixture_training(
        n_samples=_args.mixture_samples,
        max_components=_args.mixture_max_components,
        noise_std=_args.mixture_noise_std,
        epochs=_args.mixture_epochs,
        batch_size=_args.mixture_batch_size,
        threshold=_args.mixture_threshold,
        auto_threshold=_args.mixture_auto_threshold,
        threshold_grid_min=_args.mixture_threshold_grid_min,
        threshold_grid_max=_args.mixture_threshold_grid_max,
        threshold_grid_steps=_args.mixture_threshold_grid_steps,
        threshold_mode=_args.mixture_threshold_mode,
        threshold_objective=_args.mixture_threshold_objective,
        decision_rule=_args.mixture_decision_rule,
        cardinality_gap=_args.mixture_cardinality_gap,
        eval_only=_args.mixture_eval_only,
        resume_path=_args.mixture_resume_path,
        min_component_weight=_args.mixture_min_component_weight,
        skip_posthoc=_args.mixture_skip_posthoc
    )
    sys.exit(0)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = int(max(0, _args.single_seed))
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

for d in ('outputs/figures', 'outputs/logs', 'outputs/model'):
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & prepare data
# ─────────────────────────────────────────────────────────────────────────────

def parse_list(s):
    return [float(v) for v in s.strip('[]').split(', ')]


def parse_int_list(s):
    """Parse a string like '[-2,0,2]' or '[ -2, 0, 2 ]' into ints robustly."""
    s = str(s).strip()
    s = s.strip('[]')
    if s == '':
        return []
    parts = [p.strip() for p in s.split(',')]
    return [int(float(p)) for p in parts]

spectra_df = pd.read_csv(
    'ramanbiolib/db/raman_spectra_db.csv',
    converters={'wavenumbers': parse_list, 'intensity': parse_list}
)
meta_df = pd.read_csv('ramanbiolib/db/metadata_db.csv')

meta_unique = meta_df[['id', 'type']].drop_duplicates(subset='id')
df = spectra_df.merge(meta_unique, on='id')
df['class'] = df['type'].str.split('/').str[0]

print('Full dataset:', df.shape)
print(df['class'].value_counts().to_string(), '\n')

# ── Class distribution plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
counts = df['class'].value_counts()
ax.bar(counts.index, counts.values,
       color=plt.cm.tab10(np.linspace(0, 1, len(counts))))
ax.set_xlabel('Molecular Class', fontsize=12)
ax.set_ylabel('Number of Spectra', fontsize=12)
ax.set_title('Class Distribution in ramanbiolib Spectra DB', fontsize=14)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('outputs/figures/class_distribution.png', dpi=150)
plt.close()
print('Saved: outputs/figures/class_distribution.png')

# ── Filter to top-6 classes ───────────────────────────────────────────────────
KEEP_CLASSES = ['Proteins', 'Lipids', 'Saccharides',
                'AminoAcids', 'PrimaryMetabolites', 'NucleicAcids']

df_filt = df[df['class'].isin(KEEP_CLASSES)].reset_index(drop=True)
print('Filtered dataset:', df_filt.shape)
print(df_filt['class'].value_counts().to_string(), '\n')

X_raw       = np.array(df_filt['intensity'].tolist(),   dtype=np.float32)
wavenumbers = np.array(df_filt['wavenumbers'].iloc[0],  dtype=np.float32)

le          = LabelEncoder()
y_raw       = le.fit_transform(df_filt['class'])
CLASS_NAMES = list(le.classes_)
N_CLASSES   = len(CLASS_NAMES)
SEQ_LEN     = X_raw.shape[1]

print(f'Spectrum length  : {SEQ_LEN} points')
print(f'Wavenumber range : {wavenumbers[0]:.0f}-{wavenumbers[-1]:.0f} cm^-1')
print(f'Classes ({N_CLASSES})     : {CLASS_NAMES}\n')

class_prototypes = {
    class_name: X_raw[y_raw == idx].mean(axis=0)
    for idx, class_name in enumerate(CLASS_NAMES)
}
CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
PAIR_PENALTY_LAMBDA = float(max(0.0, _args.single_pair_penalty_lambda))
PAIR_PENALTY_START_EPOCH = int(max(1, _args.single_pair_penalty_start_epoch))
PAIR_PENALTY_RAMP_EPOCHS = int(max(0, _args.single_pair_penalty_ramp_epochs))
PAIR_PENALTY_CONFIDENCE_THRESHOLD = float(max(0.0, _args.single_pair_penalty_confidence_threshold))
FORBIDDEN_MASS_LAMBDA = float(max(0.0, _args.single_forbidden_mass_lambda))
FORBIDDEN_MASS_POWER = float(max(0.0, _args.single_forbidden_mass_power))
FORBIDDEN_MASS_CLASS_NAMES = [name for name in ('Saccharides', 'Lipids') if name in CLASS_TO_IDX]
FORBIDDEN_MASS_CLASS_INDICES = [CLASS_TO_IDX[name] for name in FORBIDDEN_MASS_CLASS_NAMES]
FORBIDDEN_MASS_TARGET_CLASS = 'PrimaryMetabolites' if 'PrimaryMetabolites' in CLASS_TO_IDX else None
FORBIDDEN_MASS_TARGET_IDX = CLASS_TO_IDX.get(FORBIDDEN_MASS_TARGET_CLASS) if FORBIDDEN_MASS_TARGET_CLASS else None
PAIR_PENALTY_MATRIX_NP, PAIR_PENALTY_ROWS = parse_pair_penalty_specs(
    _args.single_pair_penalty,
    CLASS_TO_IDX
)
if PAIR_PENALTY_ROWS:
    pair_penalty_df = pd.DataFrame(PAIR_PENALTY_ROWS)
    pair_penalty_df.to_csv('outputs/logs/pair_penalties.csv', index=False)
    print('Pair penalties:')
    print(pair_penalty_df.to_string(index=False))
    print(f'Soft confidence gate threshold: {PAIR_PENALTY_CONFIDENCE_THRESHOLD:.2f}')
    print()

if FORBIDDEN_MASS_LAMBDA > 0 and FORBIDDEN_MASS_CLASS_NAMES and FORBIDDEN_MASS_TARGET_CLASS:
    print('Forbidden-mass penalty:')
    print(f'  true classes: {", ".join(FORBIDDEN_MASS_CLASS_NAMES)}')
    print(f'  target class: {FORBIDDEN_MASS_TARGET_CLASS}')
    print(f'  lambda: {FORBIDDEN_MASS_LAMBDA:.4f}')
    print(f'  power: {FORBIDDEN_MASS_POWER:.2f}')
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data augmentation
# ─────────────────────────────────────────────────────────────────────────────

AUG_FACTOR = int(max(0, _args.single_aug_factor))
SHIFT_MAX = int(max(0, _args.single_shift_max))
SCALE_MIN = float(_args.single_scale_min)
SCALE_MAX = float(_args.single_scale_max)
GAUSSIAN_NOISE_STD = float(max(0.0, _args.single_gaussian_noise_std))
POISSON_NOISE_SCALE = float(max(0.0, _args.single_poisson_noise_scale))
BASELINE_WARP_SCALE = float(max(0.0, _args.single_baseline_warp_scale))
BROADENING_SIGMA_MAX = float(max(0.0, _args.single_broadening_sigma_max))
if SCALE_MAX < SCALE_MIN:
    SCALE_MIN, SCALE_MAX = SCALE_MAX, SCALE_MIN


def augment_spectra(
    X,
    y,
    factor=AUG_FACTOR,
    gaussian_noise_std=GAUSSIAN_NOISE_STD,
    scale_range=(SCALE_MIN, SCALE_MAX),
    shift_max=SHIFT_MAX,
    poisson_noise_scale=POISSON_NOISE_SCALE,
    baseline_warp_scale=BASELINE_WARP_SCALE,
    broadening_sigma_max=BROADENING_SIGMA_MAX
):
    axis = np.linspace(-1.0, 1.0, X.shape[1], dtype=np.float32)
    X_aug, y_aug = [X.copy()], [y.copy()]
    rng = np.random.default_rng(SEED)
    for _ in range(factor):
        X_new = np.empty_like(X)
        for i, spectrum in enumerate(X):
            aug = spectrum.copy()
            if shift_max > 0:
                shift = int(rng.integers(-shift_max, shift_max + 1))
                aug = shift_with_edge_padding(aug, shift)

            sigma = float(rng.uniform(0.0, broadening_sigma_max))
            if sigma > 1e-6:
                aug = gaussian_filter1d(aug, sigma=sigma, mode='nearest')

            aug = aug * float(rng.uniform(*scale_range))

            if baseline_warp_scale > 0:
                baseline = (
                    rng.normal(0.0, baseline_warp_scale * 0.35)
                    + rng.normal(0.0, baseline_warp_scale * 0.55) * axis
                    + rng.normal(0.0, baseline_warp_scale * 0.55) * (axis ** 2 - 0.33)
                )
                aug = aug + baseline.astype(np.float32)

            if poisson_noise_scale > 0:
                aug = aug + rng.normal(
                    0.0,
                    poisson_noise_scale * np.sqrt(np.clip(aug, 0.0, None) + 1e-6),
                    size=aug.shape
                ).astype(np.float32)

            if gaussian_noise_std > 0:
                aug = aug + rng.normal(0.0, gaussian_noise_std, size=aug.shape).astype(np.float32)

            X_new[i] = np.clip(aug, 0.0, None).astype(np.float32)
        X_aug.append(X_new)
        y_aug.append(y.copy())
    return np.vstack(X_aug), np.concatenate(y_aug)

X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X_raw, y_raw, test_size=0.25, stratify=y_raw, random_state=SEED
)
print(f'Raw split -> Train: {len(X_train_raw)}  Test (untouched): {len(X_test)}')

val_size = float(np.clip(_args.single_val_size, 0.05, 0.40))
X_train_core_raw, X_val, y_train_core_raw, y_val = train_test_split(
    X_train_raw, y_train_raw, test_size=val_size, stratify=y_train_raw, random_state=SEED
)
print(f'Train/Val split inside 75% -> Train: {len(X_train_core_raw)}  Val: {len(X_val)}')

class_counts_df = save_split_class_distribution(
    split_map={
        'full_dataset': y_raw,
        'train_core': y_train_core_raw,
        'val': y_val,
        'test': y_test
    },
    class_names=CLASS_NAMES,
    fig_path='outputs/figures/class_distribution_splits.png',
    csv_path='outputs/logs/class_counts.csv'
)
singleton_df = class_counts_df[
    (class_counts_df['split'] == 'train_core') & class_counts_df['is_near_singleton']
]
if len(singleton_df):
    print('Near-singleton classes in train_core:')
    print(singleton_df[['class', 'count']].to_string(index=False))
    print()

# Augment only train split to avoid train/test leakage.
X_train, y_train = augment_spectra(X_train_core_raw, y_train_core_raw)
print(f'After train-only augmentation: {X_train.shape[0]} spectra\n')

# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset / DataLoader
# ─────────────────────────────────────────────────────────────────────────────

print(
    f'Train (augmented): {len(X_train)}  '
    f'Val (raw): {len(X_val)}  Test (raw holdout): {len(X_test)}\n'
)


class RamanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


train_ds = RamanDataset(X_train, y_train)
val_ds   = RamanDataset(X_val, y_val)
test_ds  = RamanDataset(X_test, y_test)

class_counts = np.bincount(y_train, minlength=N_CLASSES)
weights      = 1.0 / class_counts[y_train]
sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

BATCH        = int(max(1, _args.single_batch_size))
if bool(int(_args.single_balanced_sampler)):
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler)
else:
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

CONV_DROPOUT = float(np.clip(_args.single_conv_dropout, 0.0, 0.9))
DENSE_DROPOUT = float(np.clip(_args.single_dense_dropout, 0.0, 0.9))

# ─────────────────────────────────────────────────────────────────────────────
# 4. 1D CNN
# ─────────────────────────────────────────────────────────────────────────────

class RamanCNN1D(nn.Module):
    """Three-block 1D CNN.  Input: (batch, 1, L)."""
    def __init__(self, input_len=1351, n_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1,  48, 15, padding=7), nn.BatchNorm1d(48),  nn.ReLU(),
            nn.Conv1d(48, 48, 15, padding=7), nn.BatchNorm1d(48),  nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(CONV_DROPOUT)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(48, 96, 11, padding=5), nn.BatchNorm1d(96),  nn.ReLU(),
            nn.Conv1d(96, 96, 11, padding=5), nn.BatchNorm1d(96),  nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(CONV_DROPOUT)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(96, 192, 7, padding=3), nn.BatchNorm1d(192), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(CONV_DROPOUT)
        )
        dummy = torch.zeros(1, 1, input_len)
        flat  = self._fwd(dummy).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(DENSE_DROPOUT),
            nn.Linear(256, n_classes)
        )

    def _fwd(self, x):
        return self.block3(self.block2(self.block1(x))).view(x.size(0), -1)

    def forward(self, x):
        return self.classifier(self._fwd(x))


model    = RamanCNN1D(input_len=SEQ_LEN, n_classes=N_CLASSES).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model parameters: {n_params:,}\n')


def build_single_label_criterion(class_weights_np, use_class_weights=True):
    class_weight_tensor = (
        torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE)
        if use_class_weights
        else None
    )
    if _args.single_loss == 'focal':
        base_loss = WeightedFocalLoss(alpha=class_weight_tensor, gamma=_args.single_focal_gamma)
    else:
        base_loss = nn.CrossEntropyLoss(weight=class_weight_tensor)

    use_pair_penalty = bool(PAIR_PENALTY_ROWS)
    use_forbidden_mass = (
        FORBIDDEN_MASS_LAMBDA > 0
        and FORBIDDEN_MASS_TARGET_IDX is not None
        and bool(FORBIDDEN_MASS_CLASS_INDICES)
    )
    if use_pair_penalty or use_forbidden_mass:
        pair_penalty_tensor = torch.tensor(PAIR_PENALTY_MATRIX_NP, dtype=torch.float32, device=DEVICE)
        return ConfusionPairPenaltyLoss(
            base_loss=base_loss,
            pair_penalty_matrix=pair_penalty_tensor,
            pair_penalty_lambda=PAIR_PENALTY_LAMBDA,
            confidence_threshold=PAIR_PENALTY_CONFIDENCE_THRESHOLD,
            forbidden_mass_target_idx=FORBIDDEN_MASS_TARGET_IDX,
            forbidden_mass_class_indices=FORBIDDEN_MASS_CLASS_INDICES,
            forbidden_mass_lambda=FORBIDDEN_MASS_LAMBDA,
            forbidden_mass_power=FORBIDDEN_MASS_POWER
        )
    return base_loss

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS = int(max(1, _args.single_epochs))
LR = float(_args.single_lr)
WD = float(_args.single_weight_decay)
PATIENCE = int(max(1, _args.single_early_stop_patience))
EARLY_STOP_MIN_EPOCH = int(max(1, _args.single_early_stop_min_epoch))
BEST_MODEL_MIN_EPOCH = int(max(1, _args.single_best_model_min_epoch))
K_FOLDS = int(max(1, _args.single_kfolds))
RESUME_SINGLE = bool(_args.single_continue_from_best)
if RESUME_SINGLE and int(_args.single_extra_epochs) > 0:
    EPOCHS = int(_args.single_extra_epochs)

cv_rows = []
if K_FOLDS > 1:
    print(f'\nRunning {K_FOLDS}-fold CV on the 75% train split...')
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train_raw, y_train_raw), start=1):
        X_fold_train_raw = X_train_raw[tr_idx]
        y_fold_train_raw = y_train_raw[tr_idx]
        X_fold_val = X_train_raw[va_idx]
        y_fold_val = y_train_raw[va_idx]
        X_fold_train, y_fold_train = augment_spectra(X_fold_train_raw, y_fold_train_raw)

        fold_train_ds = RamanDataset(X_fold_train, y_fold_train)
        fold_val_ds = RamanDataset(X_fold_val, y_fold_val)
        fold_counts = np.bincount(y_fold_train, minlength=N_CLASSES)
        fold_weights = 1.0 / np.maximum(fold_counts[y_fold_train], 1.0)
        fold_sampler = WeightedRandomSampler(fold_weights, len(fold_weights), replacement=True)
        fold_train_loader = DataLoader(fold_train_ds, batch_size=BATCH, sampler=fold_sampler)
        fold_val_loader = DataLoader(fold_val_ds, batch_size=BATCH, shuffle=False)

        fold_model = RamanCNN1D(input_len=SEQ_LEN, n_classes=N_CLASSES).to(DEVICE)
        fold_core_counts = np.bincount(y_fold_train_raw, minlength=N_CLASSES).astype(np.float32)
        fold_class_weights = fold_core_counts.sum() / (N_CLASSES * np.maximum(fold_core_counts, 1.0))
        fold_class_weights = fold_class_weights / fold_class_weights.mean()
        fold_criterion = build_single_label_criterion(fold_class_weights)
        fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=LR, weight_decay=WD)
        fold_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fold_optimizer, T_max=EPOCHS)
        fold_ema = ExponentialMovingAverage(fold_model, decay=0.995)

        fold_best_acc = 0.0
        fold_best_loss = float('inf')
        fold_best_epoch = 0
        fold_no_improve = 0
        fold_epochs_ran = 0

        for epoch in range(1, EPOCHS + 1):
            fold_pair_scale = get_pair_penalty_scale(
                epoch,
                PAIR_PENALTY_START_EPOCH,
                PAIR_PENALTY_RAMP_EPOCHS
            )
            set_loss_pair_penalty_scale(fold_criterion, fold_pair_scale)
            fold_model.train()
            for xb, yb in fold_train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    fold_optimizer.zero_grad()
                    # MixUp augmentation in-batch (pairwise mix)
                    use_mix = (_args.mixup_prob > 0 and _args.mixup_alpha > 0 and np.random.rand() < float(_args.mixup_prob))
                    if use_mix:
                        lam = float(np.random.beta(_args.mixup_alpha, _args.mixup_alpha))
                        idx_shuffle = torch.randperm(xb.size(0), device=xb.device)
                        xb2 = xb[idx_shuffle]
                        yb2 = yb[idx_shuffle]
                        xb_mixed = lam * xb + (1.0 - lam) * xb2
                        logits = fold_model(xb_mixed)
                        loss = lam * fold_criterion(logits, yb) + (1.0 - lam) * fold_criterion(logits, yb2)
                    else:
                        logits = fold_model(xb)
                        loss = fold_criterion(logits, yb)
                    loss.backward()
                    fold_optimizer.step()
                    # update EMA shadow weights
                    fold_ema.update(fold_model)
            fold_scheduler.step()

            fold_model.eval()
            v_loss = v_correct = v_total = 0
            with torch.no_grad():
                for xb, yb in fold_val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = fold_model(xb)
                    loss = fold_criterion(logits, yb)
                    v_loss += loss.item() * len(yb)
                    v_correct += (logits.argmax(1) == yb).sum().item()
                    v_total += len(yb)

            fold_epochs_ran = epoch
            v_loss_epoch = v_loss / max(v_total, 1)
            v_acc = v_correct / max(v_total, 1)
            improved = (
                (v_loss_epoch < fold_best_loss - 1e-6) or
                (abs(v_loss_epoch - fold_best_loss) <= 1e-6 and v_acc > fold_best_acc + 1e-6)
            )
            if improved:
                fold_best_loss = v_loss_epoch
                fold_best_acc = v_acc
                fold_best_epoch = epoch
                fold_no_improve = 0
            else:
                fold_no_improve += 1
            if epoch >= EARLY_STOP_MIN_EPOCH and fold_no_improve >= PATIENCE:
                break

        cv_rows.append({
            'fold': fold_idx,
            'best_val_loss': float(fold_best_loss),
            'best_val_acc': float(fold_best_acc),
            'best_epoch': int(fold_best_epoch),
            'epochs_ran': int(fold_epochs_ran),
            'train_samples_raw': int(len(X_fold_train_raw)),
            'val_samples_raw': int(len(X_fold_val))
        })
        print(
            f"CV fold {fold_idx}/{K_FOLDS}: "
            f"best_val_loss={fold_best_loss:.4f} best_val_acc={fold_best_acc:.4f} "
            f"(best_epoch={fold_best_epoch}, epochs_ran={fold_epochs_ran})"
        )

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv('outputs/logs/single_kfold_cv.csv', index=False)
    print('Saved: outputs/logs/single_kfold_cv.csv')
    print(
        f"CV val acc mean+/-std: {cv_df['best_val_acc'].mean():.4f}+/-{cv_df['best_val_acc'].std(ddof=0):.4f} | "
        f"val loss mean+/-std: {cv_df['best_val_loss'].mean():.4f}+/-{cv_df['best_val_loss'].std(ddof=0):.4f}"
    )

# Inverse-frequency class weights to improve minority-class recall.
train_core_counts = np.bincount(y_train_core_raw, minlength=N_CLASSES).astype(np.float32)
class_weights = train_core_counts.sum() / (N_CLASSES * np.maximum(train_core_counts, 1.0))
class_weights = class_weights / class_weights.mean()
criterion = build_single_label_criterion(class_weights, use_class_weights=bool(int(_args.single_class_weights)))
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
EMA_DECAY = float(np.clip(_args.single_ema_decay, 0.0, 0.99999))
ema = ExponentialMovingAverage(model, decay=EMA_DECAY)


def _eval_val(m):
    """Return (val_loss, val_acc) for model m on the val_loader."""
    m.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = m(xb)
            loss_sum += criterion(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return (loss_sum / max(total, 1)), (correct / max(total, 1))

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
history['pair_penalty_activation_rate'] = []
history['pair_penalty_mean_gate_weight'] = []
history['forbidden_mass_mean_prob'] = []
history['forbidden_mass_mean_term'] = []
best_val_acc = 0.0
best_val_loss = float('inf')
best_ckpt_path = 'outputs/model/best_model.pt'
epochs_no_improve = 0

if RESUME_SINGLE:
    resume_path = _args.single_resume_path
    if os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=DEVICE))
        print(f'Resuming single-model training from: {resume_path}')
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * len(yb)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += len(yb)
        best_val_loss = (v_loss / v_total) if v_total else float('inf')
        best_val_acc = (v_correct / v_total) if v_total else 0.0
        print(f'Initial val loss/acc from checkpoint: {best_val_loss:.4f}/{best_val_acc:.4f}')
    else:
        print(f'Warning: resume checkpoint not found, starting fresh: {resume_path}')

for epoch in range(1, EPOCHS + 1):
    pair_scale = get_pair_penalty_scale(
        epoch,
        PAIR_PENALTY_START_EPOCH,
        PAIR_PENALTY_RAMP_EPOCHS
    )
    set_loss_pair_penalty_scale(criterion, pair_scale)
    model.train()
    t_loss = t_correct = t_total = 0
    epoch_pair_activation_rates = []
    epoch_pair_gate_weights = []
    epoch_forbidden_mass_probs = []
    epoch_forbidden_mass_terms = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        # Optional MixUp: combine two examples in-batch and combine losses
        use_mix = (_args.mixup_prob > 0 and _args.mixup_alpha > 0 and np.random.rand() < float(_args.mixup_prob))
        if use_mix:
            lam = float(np.random.beta(_args.mixup_alpha, _args.mixup_alpha))
            idx_shuffle = torch.randperm(xb.size(0), device=xb.device)
            xb2 = xb[idx_shuffle]
            yb2 = yb[idx_shuffle]
            xb_mixed = lam * xb + (1.0 - lam) * xb2
            logits = model(xb_mixed)
            loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb2)
            # pair/forbidden stats: average weighted if available
            if hasattr(criterion, 'pair_penalty_stats'):
                s1 = criterion.pair_penalty_stats(logits, yb)
                s2 = criterion.pair_penalty_stats(logits, yb2)
                if s1 is not None and s2 is not None:
                    epoch_pair_activation_rates.append(lam * s1['activation_rate'] + (1 - lam) * s2['activation_rate'])
                    epoch_pair_gate_weights.append(lam * s1['mean_gate_weight'] + (1 - lam) * s2['mean_gate_weight'])
            if hasattr(criterion, 'forbidden_mass_stats'):
                f1 = criterion.forbidden_mass_stats(logits, yb)
                f2 = criterion.forbidden_mass_stats(logits, yb2)
                if f1 is not None and f2 is not None:
                    epoch_forbidden_mass_probs.append(lam * f1['mean_prob'] + (1 - lam) * f2['mean_prob'])
                    epoch_forbidden_mass_terms.append(lam * f1['mean_term'] + (1 - lam) * f2['mean_term'])
        else:
            logits = model(xb)
            loss   = criterion(logits, yb)
            if hasattr(criterion, 'pair_penalty_stats'):
                pair_stats = criterion.pair_penalty_stats(logits, yb)
                if pair_stats is not None:
                    epoch_pair_activation_rates.append(pair_stats['activation_rate'])
                    epoch_pair_gate_weights.append(pair_stats['mean_gate_weight'])
            if hasattr(criterion, 'forbidden_mass_stats'):
                forbidden_stats = criterion.forbidden_mass_stats(logits, yb)
                if forbidden_stats is not None:
                    epoch_forbidden_mass_probs.append(forbidden_stats['mean_prob'])
                    epoch_forbidden_mass_terms.append(forbidden_stats['mean_term'])
        loss.backward()
        optimizer.step()
        t_loss    += loss.item() * len(yb)
        t_correct += (logits.argmax(1) == yb).sum().item()
        t_total   += len(yb)
    scheduler.step()

    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = criterion(logits, yb)
            v_loss    += loss.item() * len(yb)
            v_correct += (logits.argmax(1) == yb).sum().item()
            v_total   += len(yb)

    t_acc = t_correct / t_total
    v_acc = v_correct / v_total
    v_loss_epoch = v_loss / v_total
    history['train_loss'].append(t_loss / t_total)
    history['train_acc'].append(t_acc)
    history['val_loss'].append(v_loss_epoch)
    history['val_acc'].append(v_acc)
    history['pair_penalty_activation_rate'].append(
        float(np.mean(epoch_pair_activation_rates)) if epoch_pair_activation_rates else 0.0
    )
    history['pair_penalty_mean_gate_weight'].append(
        float(np.mean(epoch_pair_gate_weights)) if epoch_pair_gate_weights else 0.0
    )
    history['forbidden_mass_mean_prob'].append(
        float(np.mean(epoch_forbidden_mass_probs)) if epoch_forbidden_mass_probs else 0.0
    )
    history['forbidden_mass_mean_term'].append(
        float(np.mean(epoch_forbidden_mass_terms)) if epoch_forbidden_mass_terms else 0.0
    )

    improved = (
        (v_loss_epoch < best_val_loss - 1e-6) or
        (abs(v_loss_epoch - best_val_loss) <= 1e-6 and v_acc > best_val_acc + 1e-6)
    )
    if improved:
        best_val_loss = v_loss_epoch
        best_val_acc = v_acc
        # Candidate 1: raw weights (already evaluated as v_loss_epoch / v_acc).
        # Candidate 2: EMA-shadow weights. EMA can generalise better, but with
        # short training / high decay the shadow can still be near init, so we
        # only keep it when it is genuinely no worse than the raw weights on val.
        ema.store(model)
        ema.copy_to(model)
        ema_loss, ema_acc = _eval_val(model)
        ema.restore(model)
        ema_better = (ema_acc > v_acc + 1e-6) or (
            abs(ema_acc - v_acc) <= 1e-6 and ema_loss < v_loss_epoch - 1e-6
        )
        if ema_better:
            ema.store(model)
            ema.copy_to(model)
            torch.save(model.state_dict(), best_ckpt_path)
            ema.restore(model)
        else:
            torch.save(model.state_dict(), best_ckpt_path)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{EPOCHS}  '
              f'train_loss={t_loss/t_total:.4f} acc={t_acc:.3f}  '
              f'val_loss={v_loss_epoch:.4f} acc={v_acc:.3f}  '
              f'pair_scale={pair_scale:.2f}  '
              f'pair_active={history["pair_penalty_activation_rate"][-1]:.3f}  '
              f'pair_gate_mean={history["pair_penalty_mean_gate_weight"][-1]:.4f}  '
              f'forbidden_prob={history["forbidden_mass_mean_prob"][-1]:.4f}  '
              f'forbidden_term={history["forbidden_mass_mean_term"][-1]:.4f}')

    if epoch >= EARLY_STOP_MIN_EPOCH and epochs_no_improve >= PATIENCE:
        print(
            f'Early stopping at epoch {epoch} '
            f'(no val improvement for {PATIENCE} epochs).'
        )
        break

print(f'\nBest val loss/acc: {best_val_loss:.4f}/{best_val_acc:.4f}')
pd.DataFrame(history).to_csv('outputs/logs/training_log.csv', index=False)
print('Saved: outputs/logs/training_log.csv')

# ── Training curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train_loss'], lw=2, label='Train')
axes[0].plot(history['val_loss'],   lw=2, label='Val')
axes[0].set(
    xlabel='Epoch',
    ylabel='Loss',
    title=('Weighted Focal Loss' if _args.single_loss == 'focal' else 'Weighted Cross-Entropy Loss')
)
axes[0].legend()
axes[1].plot(history['train_acc'], lw=2, label='Train')
axes[1].plot(history['val_acc'],   lw=2, label='Val')
axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Classification Accuracy')
axes[1].legend()
plt.tight_layout()
plt.savefig('outputs/figures/training_curves.png', dpi=150)
plt.close()
print('Saved: outputs/figures/training_curves.png')

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

model.load_state_dict(torch.load('outputs/model/best_model.pt', map_location=DEVICE))
model.eval()

# Optionally use Test-Time Augmentation (small shifts) to boost accuracy cheaply
all_preds, all_labels, all_probs = [], [], []
tta_shifts = parse_int_list(_args.single_tta_shifts) if _args.single_tta else [0]
with torch.no_grad():
    for xb, yb in test_loader:
        xb_np = xb.cpu().numpy().squeeze(1)  # (B, L)
        probs_accum = None
        n_shifts = len(tta_shifts) if _args.single_tta else 1
        for shift in tta_shifts:
            if shift == 0:
                xb_shift = xb.to(DEVICE)
            else:
                xb_shifted = np.stack([shift_with_edge_padding(xi, int(shift)) for xi in xb_np]).astype(np.float32)
                xb_shift = torch.tensor(xb_shifted, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            logits = model(xb_shift)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_accum = probs if probs_accum is None else probs_accum + probs
        probs_mean = probs_accum / float(n_shifts)
        all_preds.extend(probs_mean.argmax(axis=1))
        all_labels.extend(yb.numpy())
        all_probs.extend(probs_mean)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
test_acc   = accuracy_score(all_labels, all_preds)

print(f'\nTest accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)\n')
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, labels=np.arange(N_CLASSES), zero_division=0
)
per_class_metrics_df = pd.DataFrame({
    'class': CLASS_NAMES,
    'precision': np.round(precision, 4),
    'recall': np.round(recall, 4),
    'f1_score': np.round(f1, 4),
    'support': support.astype(int)
}).sort_values(['recall', 'support'], ascending=[True, True])
per_class_metrics_df.to_csv('outputs/logs/per_class_metrics.csv', index=False)
print('Saved: outputs/logs/per_class_metrics.csv')

test_prob_df = pd.DataFrame({
    'sample_idx': np.arange(len(all_labels)),
    'true_class': [CLASS_NAMES[int(idx)] for idx in all_labels],
    'predicted_class': [CLASS_NAMES[int(idx)] for idx in all_preds],
    'predicted_confidence': np.max(all_probs, axis=1),
})
for class_idx, class_name in enumerate(CLASS_NAMES):
    test_prob_df[f'p_{class_name}'] = all_probs[:, class_idx]
test_prob_df.to_csv('outputs/logs/test_predictions.csv', index=False)
print('Saved: outputs/logs/test_predictions.csv')

baseline_df = None
if not _args.single_skip_baselines:
    print('\nTraining baseline models on raw 75/25 split...')
    X_train_flat = X_train_raw.reshape(len(X_train_raw), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    baseline_models = {
        'logreg_l2': make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight='balanced',
                solver='lbfgs'
            )
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=500,
            random_state=SEED,
            class_weight='balanced_subsample',
            n_jobs=-1
        ),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=400,
            random_state=SEED
        )
    }
    baseline_rows = []
    for name, clf in baseline_models.items():
        clf.fit(X_train_flat, y_train_raw)
        pred = clf.predict(X_test_flat)
        baseline_rows.append({
            'model': name,
            'test_acc': float(accuracy_score(y_test, pred)),
            'macro_f1': float(f1_score(y_test, pred, average='macro', zero_division=0))
        })
    baseline_df = pd.DataFrame(baseline_rows).sort_values('test_acc', ascending=False)
    baseline_df.to_csv('outputs/logs/baseline_models.csv', index=False)
    print('Saved: outputs/logs/baseline_models.csv')
    print(baseline_df.to_string(index=False))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm_vals = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm_vals, display_labels=CLASS_NAMES).plot(
    ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix – Test Set', fontsize=13)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('outputs/figures/confusion_matrix.png', dpi=150)
plt.close()
print('Saved: outputs/figures/confusion_matrix.png')
top_confusions_df = save_confusion_audit(
    cm=cm_vals,
    class_names=CLASS_NAMES,
    confusion_csv_path='outputs/logs/confusion_matrix.csv',
    top_confusions_csv_path='outputs/logs/top_confusions.csv'
)
print('Saved: outputs/logs/confusion_matrix.csv')
print('Saved: outputs/logs/top_confusions.csv')

if _args.single_skip_posthoc:
    print('Skipping posthoc artifacts for fast sweep mode.')
    sys.exit(0)

save_class_prototype_plots(class_prototypes, wavenumbers, top_confusions_df)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Integrated Gradients
# ─────────────────────────────────────────────────────────────────────────────

def integrated_gradients(model, x, target_class, n_steps=50):
    """
    Compute Integrated Gradients attribution for a single spectrum.
    x : Tensor (1, 1, L)  on DEVICE
    Returns np.ndarray (L,)
    """
    baseline     = torch.zeros_like(x)
    alphas       = torch.linspace(0, 1, n_steps, device=DEVICE)
    interpolated = torch.stack(
        [baseline + a * (x - baseline) for a in alphas]
    ).squeeze(1)            # (n_steps, 1, L)
    interpolated.requires_grad_(True)
    logits = model(interpolated)
    logits[:, target_class].sum().backward()
    avg_grads = interpolated.grad.mean(dim=0)   # (1, L)
    ig = ((x - baseline).squeeze() * avg_grads.squeeze()).detach().cpu().numpy()
    return ig


def class_mean_saliency(model, X_cls, label, n_samples=15, n_steps=50):
    model.eval()
    igs = []
    idx = np.random.choice(len(X_cls), min(n_samples, len(X_cls)), replace=False)
    for i in idx:
        x  = torch.tensor(X_cls[i]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        ig = integrated_gradients(model, x, label, n_steps=n_steps)
        igs.append(np.abs(ig))
    return np.mean(igs, axis=0)


def save_wrong_prediction_audit(
    model,
    x_test,
    y_true,
    y_pred,
    y_prob,
    class_names,
    prototypes,
    wavenumbers,
    max_examples=50,
    page_size=10
):
    old_pages = [
        os.path.join('outputs/figures', name)
        for name in os.listdir('outputs/figures')
        if name.startswith('wrong_predictions_page_') and name.endswith('.png')
    ]
    for old_page in old_pages:
        os.remove(old_page)

    wrong_idx = np.flatnonzero(y_true != y_pred)
    if len(wrong_idx) == 0:
        pd.DataFrame(columns=[
            'rank', 'test_index', 'true_class', 'predicted_class',
            'predicted_confidence', 'top_saliency_peaks_cm'
        ]).to_csv('outputs/logs/wrong_predictions.csv', index=False)
        print('Saved: outputs/logs/wrong_predictions.csv')
        return

    confidences = y_prob[wrong_idx, y_pred[wrong_idx]]
    ordered_wrong_idx = wrong_idx[np.argsort(confidences)[::-1][:max_examples]]
    rows = []
    rendered = []

    for rank, sample_idx in enumerate(ordered_wrong_idx, start=1):
        pred_idx = int(y_pred[sample_idx])
        true_idx = int(y_true[sample_idx])
        spectrum = x_test[sample_idx]
        x_tensor = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        saliency = np.abs(integrated_gradients(model, x_tensor, pred_idx, n_steps=32))
        saliency_norm = normalize_curve(saliency)
        peaks, _ = find_peaks(saliency_norm, prominence=0.15, distance=20)
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(saliency_norm))])
        top_peaks = peaks[np.argsort(saliency_norm[peaks])[::-1][:3]]
        peak_labels = '|'.join(str(int(wavenumbers[p])) for p in top_peaks)

        rows.append({
            'rank': rank,
            'test_index': int(sample_idx),
            'true_class': class_names[true_idx],
            'predicted_class': class_names[pred_idx],
            'predicted_confidence': round(float(y_prob[sample_idx, pred_idx]), 4),
            'top_saliency_peaks_cm': peak_labels
        })
        rendered.append({
            'rank': rank,
            'sample_idx': int(sample_idx),
            'true_name': class_names[true_idx],
            'pred_name': class_names[pred_idx],
            'confidence': float(y_prob[sample_idx, pred_idx]),
            'spectrum': spectrum,
            'true_proto': prototypes[class_names[true_idx]],
            'pred_proto': prototypes[class_names[pred_idx]],
            'saliency_norm': saliency_norm,
            'top_peaks': top_peaks
        })

    wrong_df = pd.DataFrame(rows)
    wrong_df.to_csv('outputs/logs/wrong_predictions.csv', index=False)
    print('Saved: outputs/logs/wrong_predictions.csv')

    n_pages = int(math.ceil(len(rendered) / page_size))
    for page_idx in range(n_pages):
        page_items = rendered[page_idx * page_size:(page_idx + 1) * page_size]
        ncols = 2
        nrows = int(math.ceil(len(page_items) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')

        for ax, item in zip(axes.flat, page_items):
            ax.axis('on')
            ax.plot(wavenumbers, item['spectrum'], color='black', lw=1.4)
            ax.plot(wavenumbers, item['true_proto'], color='steelblue', lw=1.1, ls='--')
            ax.plot(wavenumbers, item['pred_proto'], color='darkorange', lw=1.1, ls=':')
            ax.set_title(
                f"#{item['rank']} true={item['true_name']} pred={item['pred_name']} "
                f"p={item['confidence']:.2f}",
                fontsize=10
            )
            ax.set_xlabel('Wavenumber (cm^-1)', fontsize=9)
            ax.set_ylabel('Intensity', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            ax2 = ax.twinx()
            ax2.fill_between(wavenumbers, item['saliency_norm'], color='crimson', alpha=0.20)
            ax2.set_yticks([])
            for peak_idx in item['top_peaks']:
                ax.axvline(wavenumbers[peak_idx], color='crimson', lw=0.8, ls='--', alpha=0.7)

        fig.suptitle(
            'Wrong Predictions: spectrum (black), true prototype (blue), predicted prototype (orange), IG (red)',
            fontsize=12
        )
        plt.tight_layout()
        page_path = f'outputs/figures/wrong_predictions_page_{page_idx + 1:02d}.png'
        plt.savefig(page_path, dpi=150)
        plt.close(fig)
        print(f'Saved: {page_path}')


save_wrong_prediction_audit(
    model=model,
    x_test=X_test,
    y_true=all_labels,
    y_pred=all_preds,
    y_prob=all_probs,
    class_names=CLASS_NAMES,
    prototypes=class_prototypes,
    wavenumbers=wavenumbers
)


print('\nComputing Integrated Gradient saliency maps...')
saliency_maps  = {}
class_spectra  = {}

for cls_name in CLASS_NAMES:
    cls_idx = le.transform([cls_name])[0]
    mask    = y_raw == cls_idx
    X_cls   = X_raw[mask]
    print(f'  {cls_name} ({len(X_cls)} spectra) ...', end=' ', flush=True)
    saliency_maps[cls_name] = class_mean_saliency(model, X_cls, cls_idx)
    class_spectra[cls_name] = X_cls.mean(axis=0)
    print('done')

# ─────────────────────────────────────────────────────────────────────────────
# 8. Saliency overlay plots
# ─────────────────────────────────────────────────────────────────────────────

COLORS = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))

for i, cls_name in enumerate(CLASS_NAMES):
    sal   = saliency_maps[cls_name]
    spec  = class_spectra[cls_name]
    sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(wavenumbers, spec, color=COLORS[i], lw=1.5, label='Mean spectrum')
    ax1.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Intensity (normalised)', color=COLORS[i], fontsize=12)
    ax1.tick_params(axis='y', labelcolor=COLORS[i])

    ax2 = ax1.twinx()
    ax2.fill_between(wavenumbers, sal_n, alpha=0.35, color='crimson', label='IG saliency')
    ax2.set_ylabel('Normalised |IG| saliency', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')

    peaks, _ = find_peaks(sal_n, prominence=0.15, distance=20)
    if len(peaks) > 0:
        top_pk = peaks[np.argsort(sal_n[peaks])[::-1][:3]]
        for pk in top_pk:
            ax1.axvline(wavenumbers[pk], color='grey', lw=0.8, ls='--')
            ax1.text(wavenumbers[pk] + 5, spec.max() * 0.9,
                     f'{wavenumbers[pk]:.0f}', fontsize=8, rotation=90)

    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, loc='upper right', fontsize=9)
    ax1.set_title(f'Saliency Map – {cls_name}', fontsize=14)
    plt.tight_layout()
    fp = f'outputs/figures/saliency_{cls_name.lower()}.png'
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f'Saved: {fp}')

# ── Aggregate heatmap ─────────────────────────────────────────────────────────
sal_matrix = np.array([
    (saliency_maps[c] - saliency_maps[c].min()) /
    (saliency_maps[c].max() - saliency_maps[c].min() + 1e-9)
    for c in CLASS_NAMES
])
step  = 10
wn_ds = wavenumbers[::step]
sd_ds = sal_matrix[:, ::step]

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(sd_ds, aspect='auto', cmap='hot',
               extent=[wn_ds[0], wn_ds[-1], len(CLASS_NAMES)-0.5, -0.5])
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_yticklabels(CLASS_NAMES, fontsize=11)
ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
ax.set_title('Integrated-Gradient Saliency Heatmap (all classes)', fontsize=13)
plt.colorbar(im, ax=ax, label='Normalised |IG|')
plt.tight_layout()
plt.savefig('outputs/figures/saliency_heatmap_all.png', dpi=150)
plt.close()
print('Saved: outputs/figures/saliency_heatmap_all.png')

# ─────────────────────────────────────────────────────────────────────────────
# 9. Key spectral regions summary
# ─────────────────────────────────────────────────────────────────────────────

WINDOW       = 20
summary_rows = []

for cls_name in CLASS_NAMES:
    sal   = saliency_maps[cls_name]
    sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
    peaks, _ = find_peaks(sal_n, prominence=0.10, distance=15)
    if len(peaks) == 0:
        peaks = np.array([np.argmax(sal_n)])
    ranked = peaks[np.argsort(sal_n[peaks])[::-1]]
    for pk in ranked[:5]:
        wn = wavenumbers[pk]
        summary_rows.append({
            'class': cls_name,
            'center_cm': int(wn),
            'range': f'{int(wn-WINDOW)}-{int(wn+WINDOW)} cm^-1',
            'saliency_score': round(float(sal_n[pk]), 4)
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('outputs/logs/key_spectral_regions.csv', index=False)
print('\nKey spectral regions:')
print(summary_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 10. Save artefacts
# ─────────────────────────────────────────────────────────────────────────────

torch.save(model.state_dict(), 'outputs/model/final_model.pt')
try:
    # also save EMA-weighted final weights if EMA was used
    ema
except NameError:
    pass
else:
    ema.store(model)
    ema.copy_to(model)
    torch.save(model.state_dict(), 'outputs/model/final_model_ema.pt')
    ema.restore(model)

config = {
    'input_len': SEQ_LEN,
    'n_classes': N_CLASSES,
    'class_names': CLASS_NAMES,
    'wavenumber_range': [int(wavenumbers[0]), int(wavenumbers[-1])],
    'batch_size': BATCH,
    'learning_rate': LR,
    'weight_decay': WD,
    'loss_name': _args.single_loss,
    'focal_gamma': (float(_args.single_focal_gamma) if _args.single_loss == 'focal' else None),
    'balanced_sampler': bool(int(_args.single_balanced_sampler)),
    'class_weights': bool(int(_args.single_class_weights)),
    'pair_penalty_lambda': (PAIR_PENALTY_LAMBDA if PAIR_PENALTY_ROWS else 0.0),
    'pair_penalty_start_epoch': (PAIR_PENALTY_START_EPOCH if PAIR_PENALTY_ROWS else None),
    'pair_penalty_ramp_epochs': (PAIR_PENALTY_RAMP_EPOCHS if PAIR_PENALTY_ROWS else None),
    'pair_penalty_confidence_threshold': (PAIR_PENALTY_CONFIDENCE_THRESHOLD if PAIR_PENALTY_ROWS else None),
    'pair_penalties': PAIR_PENALTY_ROWS,
    'forbidden_mass_lambda': (FORBIDDEN_MASS_LAMBDA if FORBIDDEN_MASS_LAMBDA > 0 else 0.0),
    'forbidden_mass_power': (FORBIDDEN_MASS_POWER if FORBIDDEN_MASS_LAMBDA > 0 else None),
    'forbidden_mass_classes': (FORBIDDEN_MASS_CLASS_NAMES if FORBIDDEN_MASS_LAMBDA > 0 else []),
    'forbidden_mass_target_class': (FORBIDDEN_MASS_TARGET_CLASS if FORBIDDEN_MASS_LAMBDA > 0 else None),
    'conv_dropout': CONV_DROPOUT,
    'dense_dropout': DENSE_DROPOUT,
    'augmentation_factor': AUG_FACTOR,
    'augmentation_shift_max': SHIFT_MAX,
    'augmentation_scale_range': [round(float(SCALE_MIN), 4), round(float(SCALE_MAX), 4)],
    'augmentation_gaussian_noise_std': round(float(GAUSSIAN_NOISE_STD), 6),
    'augmentation_poisson_noise_scale': round(float(POISSON_NOISE_SCALE), 6),
    'augmentation_baseline_warp_scale': round(float(BASELINE_WARP_SCALE), 6),
    'augmentation_broadening_sigma_max': round(float(BROADENING_SIGMA_MAX), 4),
    'cnn_channels': [48, 96, 192],
    'classifier_hidden': 256,
    'epochs_target': EPOCHS,
    'epochs_ran': len(history['train_loss']),
    'early_stop_min_epoch': EARLY_STOP_MIN_EPOCH,
    'k_folds': K_FOLDS,
    'kfold_cv_ran': bool(K_FOLDS > 1),
    'kfold_cv_rows': int(len(cv_rows)),
    'kfold_val_acc_mean': (round(float(np.mean([r['best_val_acc'] for r in cv_rows])), 4) if cv_rows else None),
    'kfold_val_acc_std': (round(float(np.std([r['best_val_acc'] for r in cv_rows])), 4) if cv_rows else None),
    'kfold_val_loss_mean': (round(float(np.mean([r['best_val_loss'] for r in cv_rows])), 4) if cv_rows else None),
    'kfold_val_loss_std': (round(float(np.std([r['best_val_loss'] for r in cv_rows])), 4) if cv_rows else None),
    'resume_from_checkpoint': RESUME_SINGLE,
    'val_size_within_train': val_size,
    'best_val_acc': round(best_val_acc, 4),
    'best_val_loss': round(float(best_val_loss), 4),
    'test_acc': round(float(test_acc), 4),
    'test_macro_f1': round(float(f1_score(all_labels, all_preds, average='macro', zero_division=0)), 4),
    'pair_penalty_activation_rate_mean': round(float(np.mean(history['pair_penalty_activation_rate'])), 4),
    'pair_penalty_mean_gate_weight_mean': round(float(np.mean(history['pair_penalty_mean_gate_weight'])), 6),
    'forbidden_mass_mean_prob_mean': round(float(np.mean(history['forbidden_mass_mean_prob'])), 6),
    'forbidden_mass_mean_term_mean': round(float(np.mean(history['forbidden_mass_mean_term'])), 6),
    'baselines_ran': bool(baseline_df is not None),
    'baseline_best_model': (str(baseline_df.iloc[0]['model']) if baseline_df is not None and len(baseline_df) else None),
    'baseline_best_test_acc': (round(float(baseline_df.iloc[0]['test_acc']), 4) if baseline_df is not None and len(baseline_df) else None),
    'baseline_best_macro_f1': (round(float(baseline_df.iloc[0]['macro_f1']), 4) if baseline_df is not None and len(baseline_df) else None)
}
with open('outputs/model/model_config.json', 'w') as fh:
    json.dump(config, fh, indent=2)

np.savez('outputs/logs/saliency_maps.npz',
         wavenumbers=wavenumbers,
         class_names=np.array(CLASS_NAMES),
         **{cls: saliency_maps[cls] for cls in CLASS_NAMES})

print('\n=== All artefacts saved ===')
print(f'FINAL TEST ACCURACY: {test_acc*100:.2f}%')

