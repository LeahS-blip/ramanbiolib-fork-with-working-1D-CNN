"""
ensemble_raman.py
=================
Seed-ensemble + Test-Time-Augmentation evaluator for the single-label Raman
classifier.

Why this exists
---------------
The dataset is tiny (194 spectra, ~49-sample holdout).  A single CNN sits around
~91% test accuracy and is noise-dominated at that scale.  The most reliable way
to gain a few points on small data is to (1) train several models that differ
only in their initialisation / augmentation draw, and (2) average their softmax
outputs together with a handful of small spectral shifts (TTA).

Crucially, the train/test split is held FIXED (random_state=42, identical to
train_cnn_raman.py) so every ensemble member is evaluated on exactly the same
holdout.  Only the model-init / augmentation seed varies between members, which
is what makes averaging them meaningful.

Usage:
    python ensemble_raman.py --members 9 --epochs 120
    python ensemble_raman.py --members 9 --epochs 120 --tta "[-3,-2,0,2,3]"
"""

import os, json, argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.ndimage import gaussian_filter1d

import warnings
warnings.filterwarnings('ignore')

SPLIT_SEED = 42          # MUST match train_cnn_raman.py so the holdout is identical
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '8')))
KEEP_CLASSES = ['Proteins', 'Lipids', 'Saccharides',
                'AminoAcids', 'PrimaryMetabolites', 'NucleicAcids']

# Augmentation hyper-parameters (identical to train_cnn_raman.py defaults)
AUG_FACTOR = 15
SHIFT_MAX = 4
SCALE_MIN, SCALE_MAX = 0.92, 1.08
GAUSSIAN_NOISE_STD = 0.01
POISSON_NOISE_SCALE = 0.008
BASELINE_WARP_SCALE = 0.03
BROADENING_SIGMA_MAX = 1.0


def shift_with_edge_padding(x, shift):
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


def smote_oversample(X, y, n_classes, target_per_class, seed, k_neighbors=4):
    """Spectral SMOTE: interpolate minority classes up to target_per_class.

    For each minority class, repeatedly pick a sample and one of its k nearest
    same-class neighbours and create a convex blend a*x_i + (1-a)*x_nn.  Returns
    the original data plus synthetic minority rows.  Note: blends of two distinct
    molecules are physically mixture-like, so this is offered for comparison, not
    as the default.
    """
    rng = np.random.default_rng(seed)
    X_out, y_out = [X.copy()], [y.copy()]
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        n = len(idx)
        need = target_per_class - n
        if need <= 0 or n < 2:
            continue
        Xc = X[idx]
        # pairwise euclidean distances within the class
        d = np.sqrt(((Xc[:, None, :] - Xc[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        kk = min(k_neighbors, n - 1)
        nn_idx = np.argsort(d, axis=1)[:, :kk]
        syn = np.empty((need, X.shape[1]), dtype=np.float32)
        for j in range(need):
            i = int(rng.integers(0, n))
            nb = int(nn_idx[i, int(rng.integers(0, kk))])
            a = float(rng.uniform(0.0, 1.0))
            syn[j] = (a * Xc[i] + (1 - a) * Xc[nb]).astype(np.float32)
        X_out.append(np.clip(syn, 0.0, None))
        y_out.append(np.full(need, c, dtype=y.dtype))
    return np.vstack(X_out), np.concatenate(y_out)


def augment_spectra(X, y, factor, seed):
    axis = np.linspace(-1.0, 1.0, X.shape[1], dtype=np.float32)
    X_aug, y_aug = [X.copy()], [y.copy()]
    rng = np.random.default_rng(seed)
    for _ in range(factor):
        X_new = np.empty_like(X)
        for i, spectrum in enumerate(X):
            aug = spectrum.copy()
            if SHIFT_MAX > 0:
                aug = shift_with_edge_padding(aug, int(rng.integers(-SHIFT_MAX, SHIFT_MAX + 1)))
            sigma = float(rng.uniform(0.0, BROADENING_SIGMA_MAX))
            if sigma > 1e-6:
                aug = gaussian_filter1d(aug, sigma=sigma, mode='nearest')
            aug = aug * float(rng.uniform(SCALE_MIN, SCALE_MAX))
            if BASELINE_WARP_SCALE > 0:
                baseline = (rng.normal(0.0, BASELINE_WARP_SCALE * 0.35)
                            + rng.normal(0.0, BASELINE_WARP_SCALE * 0.55) * axis
                            + rng.normal(0.0, BASELINE_WARP_SCALE * 0.55) * (axis ** 2 - 0.33))
                aug = aug + baseline.astype(np.float32)
            if POISSON_NOISE_SCALE > 0:
                aug = aug + rng.normal(0.0, POISSON_NOISE_SCALE * np.sqrt(np.clip(aug, 0.0, None) + 1e-6),
                                       size=aug.shape).astype(np.float32)
            if GAUSSIAN_NOISE_STD > 0:
                aug = aug + rng.normal(0.0, GAUSSIAN_NOISE_STD, size=aug.shape).astype(np.float32)
            X_new[i] = np.clip(aug, 0.0, None).astype(np.float32)
        X_aug.append(X_new)
        y_aug.append(y.copy())
    return np.vstack(X_aug), np.concatenate(y_aug)


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.register_buffer('alpha', alpha.float() if alpha is not None else None)
        self.gamma = float(gamma)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        target = target.long()
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp().clamp_min(1e-8)
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = self.alpha.gather(0, target) * loss
        return loss.mean()


class EMA:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach().cpu(), alpha=1 - self.decay)

    def store(self, model):
        self.backup = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n].to(p.device))

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n].to(p.device))


class RamanCNN1D(nn.Module):
    def __init__(self, input_len, n_classes, width=1.0, hidden=256,
                 conv_dropout=0.15, dense_dropout=0.15):
        super().__init__()
        c1 = max(4, int(round(48 * width)))
        c2 = max(8, int(round(96 * width)))
        c3 = max(8, int(round(192 * width)))
        self.block1 = nn.Sequential(
            nn.Conv1d(1, c1, 15, padding=7), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c1, 15, padding=7), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(conv_dropout))
        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, 11, padding=5), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.Conv1d(c2, c2, 11, padding=5), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(conv_dropout))
        self.block3 = nn.Sequential(
            nn.Conv1d(c2, c3, 7, padding=3), nn.BatchNorm1d(c3), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(conv_dropout))
        flat = self._fwd(torch.zeros(1, 1, input_len)).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(flat, hidden), nn.ReLU(), nn.Dropout(dense_dropout),
            nn.Linear(hidden, n_classes))

    def _fwd(self, x):
        return self.block3(self.block2(self.block1(x))).view(x.size(0), -1)

    def forward(self, x):
        return self.classifier(self._fwd(x))


class RamanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_data():
    def pl(s):
        return [float(v) for v in s.strip('[]').split(', ')]
    sp = pd.read_csv('ramanbiolib/db/raman_spectra_db.csv',
                     converters={'wavenumbers': pl, 'intensity': pl})
    md = pd.read_csv('ramanbiolib/db/metadata_db.csv')
    mu = md[['id', 'type']].drop_duplicates('id')
    df = sp.merge(mu, on='id')
    df['class'] = df['type'].str.split('/').str[0]
    df = df[df['class'].isin(KEEP_CLASSES)].reset_index(drop=True)
    X = np.array(df['intensity'].tolist(), dtype=np.float32)
    wn = np.array(df['wavenumbers'].iloc[0], dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df['class'])
    return X, y, wn, list(le.classes_)


def train_one_member(X_core, y_core, X_val, y_val, seq_len, n_classes,
                     class_weights, epochs, lr, wd, member_seed,
                     loss_name='cross_entropy', focal_gamma=2.0,
                     mixup_alpha=0.4, mixup_prob=0.0, ema_decay=0.99,
                     aug_factor=AUG_FACTOR, patience=15, smote_target=0,
                     width=1.0, hidden=256, batch=32, balanced=True, verbose=False):
    torch.manual_seed(member_seed)
    np.random.seed(member_seed)

    X_c, y_c = X_core, y_core
    if smote_target and smote_target > 0:
        X_c, y_c = smote_oversample(X_core, y_core, n_classes, smote_target, seed=member_seed)
    X_tr, y_tr = augment_spectra(X_c, y_c, aug_factor, seed=member_seed)
    tr_ds = RamanDataset(X_tr, y_tr)
    val_ds = RamanDataset(X_val, y_val)
    if balanced:
        counts = np.bincount(y_tr, minlength=n_classes)
        w = 1.0 / counts[y_tr]
        sampler = WeightedRandomSampler(w, len(w), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=batch, sampler=sampler)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)

    # When de-balanced, drop class weights too (uniform) so we don't over-promote minorities.
    cw_np = class_weights if balanced else np.ones(n_classes, dtype=np.float32)
    cw = torch.tensor(cw_np, dtype=torch.float32, device=DEVICE)
    model = RamanCNN1D(seq_len, n_classes, width=width, hidden=hidden).to(DEVICE)
    if loss_name == 'focal':
        crit = WeightedFocalLoss(alpha=cw, gamma=focal_gamma)
    else:
        crit = nn.CrossEntropyLoss(weight=cw)
    # Proven recipe: plain Adam + cosine annealing (matches the 91.1% baseline).
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ema = EMA(model, decay=ema_decay)

    best_state = None
    best_loss = float('inf')
    best_acc = 0.0
    no_improve = 0

    def eval_val(m):
        m.eval()
        ls = correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = m(xb)
                ls += crit(logits, yb).item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
        return ls / max(total, 1), correct / max(total, 1)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            if mixup_prob > 0 and mixup_alpha > 0 and np.random.rand() < mixup_prob:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(xb.size(0), device=xb.device)
                xb_m = lam * xb + (1 - lam) * xb[perm]
                logits = model(xb_m)
                loss = lam * crit(logits, yb) + (1 - lam) * crit(logits, yb[perm])
            else:
                loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            ema.update(model)
        sched.step()

        v_loss, v_acc = eval_val(model)
        # candidate: raw vs EMA, keep whichever is better on val
        ema.store(model); ema.copy_to(model)
        e_loss, e_acc = eval_val(model)
        use_ema = (e_acc > v_acc + 1e-6) or (abs(e_acc - v_acc) <= 1e-6 and e_loss < v_loss - 1e-6)
        if not use_ema:
            ema.restore(model)
        cur_loss, cur_acc = (e_loss, e_acc) if use_ema else (v_loss, v_acc)
        improved = (cur_loss < best_loss - 1e-6) or (abs(cur_loss - best_loss) <= 1e-6 and cur_acc > best_acc + 1e-6)
        if improved:
            best_loss, best_acc = cur_loss, cur_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if use_ema:
            ema.restore(model)
        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f'    epoch {epoch:3d}/{epochs} val_loss={v_loss:.4f} val_acc={v_acc:.3f} best_acc={best_acc:.3f}')
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_loss, best_acc


def predict_proba_tta(model, X, tta_shifts):
    model.eval()
    probs = np.zeros((len(X), 0))
    accum = None
    with torch.no_grad():
        for shift in tta_shifts:
            Xs = np.stack([shift_with_edge_padding(xi, int(shift)) for xi in X]).astype(np.float32)
            xb = torch.tensor(Xs, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            p = torch.softmax(model(xb), dim=1).cpu().numpy()
            accum = p if accum is None else accum + p
    return accum / float(len(tta_shifts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', type=int, default=7)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--val-size', type=float, default=0.15)
    ap.add_argument('--tta', type=str, default='[-3,-2,0,2,3]')
    ap.add_argument('--seed-base', type=int, default=1000)
    ap.add_argument('--loss', choices=['cross_entropy', 'focal'], default='cross_entropy')
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--mixup-prob', type=float, default=0.0)
    ap.add_argument('--mixup-alpha', type=float, default=0.4)
    ap.add_argument('--aug-factor', type=int, default=15)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--ema-decay', type=float, default=0.99)
    ap.add_argument('--smote', type=int, default=0,
                    help='If >0, SMOTE-oversample each class up to this count before augmentation')
    ap.add_argument('--width', type=float, default=1.0, help='Channel width multiplier (1.0 = 48/96/192)')
    ap.add_argument('--hidden', type=int, default=256, help='Classifier hidden units')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--balanced', type=int, choices=[0, 1], default=1,
                    help='1 = weighted sampler + class weights; 0 = plain shuffle, uniform weights')
    ap.add_argument('--out', type=str, default='outputs/logs/ensemble_results.json')
    args = ap.parse_args()

    tta_shifts = [int(float(v)) for v in args.tta.strip('[]').split(',') if v.strip() != '']

    X, y, wn, class_names = load_data()
    n_classes = len(class_names)
    seq_len = X.shape[1]

    # SAME split as train_cnn_raman.py
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SPLIT_SEED)
    X_core, X_val, y_core, y_val = train_test_split(
        X_train_raw, y_train_raw, test_size=args.val_size, stratify=y_train_raw, random_state=SPLIT_SEED)

    print(f'Device: {DEVICE}')
    print(f'Split -> core {len(X_core)}  val {len(X_val)}  test {len(X_test)}')
    print(f'Classes: {class_names}')
    print(f'TTA shifts: {tta_shifts}   Members: {args.members}\n')

    core_counts = np.bincount(y_core, minlength=n_classes).astype(np.float32)
    class_weights = core_counts.sum() / (n_classes * np.maximum(core_counts, 1.0))
    class_weights = class_weights / class_weights.mean()

    ens_test = np.zeros((len(X_test), n_classes))
    ens_val = np.zeros((len(X_val), n_classes))
    member_correct = np.zeros(len(X_test), dtype=int)   # per-sample: how many members got it right
    member_rows = []
    for m in range(args.members):
        seed = args.seed_base + m
        model, b_loss, b_acc = train_one_member(
            X_core, y_core, X_val, y_val, seq_len, n_classes, class_weights,
            args.epochs, args.lr, args.wd, member_seed=seed,
            loss_name=args.loss, focal_gamma=args.focal_gamma,
            mixup_alpha=args.mixup_alpha, mixup_prob=args.mixup_prob,
            ema_decay=args.ema_decay, aug_factor=args.aug_factor,
            patience=args.patience, smote_target=args.smote,
            width=args.width, hidden=args.hidden, batch=args.batch,
            balanced=bool(args.balanced), verbose=(m == 0))
        p_test = predict_proba_tta(model, X_test, tta_shifts)
        p_val = predict_proba_tta(model, X_val, tta_shifts)
        member_correct += (p_test.argmax(1) == y_test).astype(int)
        ens_test += p_test
        ens_val += p_val
        m_test_acc = accuracy_score(y_test, p_test.argmax(1))
        m_val_acc = accuracy_score(y_val, p_val.argmax(1))
        run_test_acc = accuracy_score(y_test, (ens_test / (m + 1)).argmax(1))
        run_val_acc = accuracy_score(y_val, (ens_val / (m + 1)).argmax(1))
        member_rows.append({'member': m, 'seed': seed, 'best_val_loss': round(b_loss, 4),
                            'member_test_acc': round(m_test_acc, 4), 'member_val_acc': round(m_val_acc, 4),
                            'running_ens_test_acc': round(run_test_acc, 4),
                            'running_ens_val_acc': round(run_val_acc, 4)})
        print(f'Member {m+1}/{args.members} (seed {seed}): test={m_test_acc:.4f} val={m_val_acc:.4f} '
              f'| ensemble so far: test={run_test_acc:.4f} val={run_val_acc:.4f}')

    ens_test /= args.members
    ens_val /= args.members
    y_pred = ens_test.argmax(1)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    val_acc = accuracy_score(y_val, ens_val.argmax(1))

    print('\n================ ENSEMBLE RESULT ================')
    print(f'Members: {args.members}  TTA: {tta_shifts}')
    print(f'Ensemble VAL  accuracy: {val_acc:.4f}')
    print(f'Ensemble TEST accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)   macro-F1: {test_f1:.4f}')
    print('\nPer-class report (test):')
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print('Confusion matrix (rows=true, cols=pred):')
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=class_names, columns=class_names).to_string())

    # Diagnostic: which test samples are HARD (few/no members ever got them right)?
    print('\nPer-sample difficulty (samples the ensemble got WRONG):')
    wrong = np.flatnonzero(y_pred != y_test)
    for i in wrong:
        print(f'  test#{i:2d} true={class_names[y_test[i]]:<18} pred={class_names[y_pred[i]]:<18} '
              f'members_correct={member_correct[i]}/{args.members}  ens_conf={ens_test[i, y_pred[i]]:.2f}')
    never = np.flatnonzero(member_correct == 0)
    print(f'\nSamples NO member ever classified correctly (hard ceiling): {len(never)} -> {list(never)}')
    print(f'If these are unfixable, max achievable test acc = {(len(y_test)-len(never))/len(y_test):.4f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'members': args.members, 'tta_shifts': tta_shifts,
                   'width': args.width, 'hidden': args.hidden, 'batch': args.batch,
                   'balanced': bool(args.balanced), 'loss': args.loss,
                   'mixup_prob': args.mixup_prob, 'smote': args.smote,
                   'aug_factor': args.aug_factor, 'epochs': args.epochs,
                   'ensemble_test_acc': round(float(test_acc), 4),
                   'ensemble_test_macro_f1': round(float(test_f1), 4),
                   'ensemble_val_acc': round(float(val_acc), 4),
                   'never_correct_idx': [int(i) for i in never],
                   'max_achievable_acc': round(float((len(y_test)-len(never))/len(y_test)), 4),
                   'class_names': class_names,
                   'member_rows': member_rows}, fh, indent=2)
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
