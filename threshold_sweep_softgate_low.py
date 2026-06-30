import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 1
THRESHOLDS = [0.02, 0.01, 0.005]
SUMMARY_DIR = Path('outputs/comparisons/20260429_softgate_threshold_sweep_low')
RUNS_DIR = SUMMARY_DIR / 'runs'
HIST_DIR = SUMMARY_DIR / 'histograms'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

hist_rows = []
summary_rows = []
for tau in THRESHOLDS:
    tau_label = f'{tau:.3f}'.replace('.', 'p')
    run_dir = RUNS_DIR / f'tau_{tau_label}'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'=== Tau {tau:.3f} ===', flush=True)
    cmd = [
        sys.executable,
        'train_cnn_raman.py',
        '--task', 'single',
        '--single-seed', str(SEED),
        '--single-epochs', '150',
        '--single-batch-size', '64',
        '--single-aug-factor', '5',
        '--single-pair-penalty', 'Saccharides:PrimaryMetabolites:3',
        '--single-pair-penalty-lambda', '0.5',
        '--single-pair-penalty-start-epoch', '81',
        '--single-pair-penalty-ramp-epochs', '70',
        '--single-pair-penalty-confidence-threshold', str(tau),
        '--single-early-stop-patience', '1000',
        '--single-early-stop-min-epoch', '150',
        '--single-skip-baselines',
        '--single-skip-posthoc',
    ]
    log_path = run_dir / 'train.log'
    with log_path.open('w', encoding='utf-8') as log_file:
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    per = pd.read_csv('outputs/logs/per_class_metrics.csv')
    cm = pd.read_csv('outputs/logs/confusion_matrix.csv')
    log = pd.read_csv('outputs/logs/training_log.csv')
    preds = pd.read_csv('outputs/logs/test_predictions.csv')

    run_prefix = f'tau_{tau_label}'
    shutil.copy2('outputs/logs/test_predictions.csv', run_dir / f'{run_prefix}_test_predictions.csv')
    shutil.copy2('outputs/logs/per_class_metrics.csv', run_dir / f'{run_prefix}_per_class_metrics.csv')
    shutil.copy2('outputs/logs/confusion_matrix.csv', run_dir / f'{run_prefix}_confusion_matrix.csv')
    shutil.copy2('outputs/logs/training_log.csv', run_dir / f'{run_prefix}_training_log.csv')

    def per_class_value(class_name: str, column: str) -> float:
        return float(per.loc[per['class'] == class_name, column].iloc[0])

    def confusion(true_name: str, pred_name: str) -> int:
        return int(cm.loc[
            (cm['true_class'] == true_name) & (cm['predicted_class'] == pred_name),
            'count'
        ].iloc[0])

    pm_probs = preds['p_PrimaryMetabolites']
    for class_name in ['Saccharides', 'Lipids', 'PrimaryMetabolites']:
        class_probs = preds.loc[preds['true_class'] == class_name, 'p_PrimaryMetabolites'].to_numpy()
        hist_rows.append({
            'tau': tau,
            'true_class': class_name,
            'count': int(len(class_probs)),
            'mean_p_pm': float(np.mean(class_probs)) if len(class_probs) else float('nan'),
            'median_p_pm': float(np.median(class_probs)) if len(class_probs) else float('nan'),
            'p90_p_pm': float(np.quantile(class_probs, 0.90)) if len(class_probs) else float('nan'),
            'frac_above_tau': float(np.mean(class_probs > tau)) if len(class_probs) else float('nan'),
        })

    summary_rows.append({
        'tau': tau,
        'seed': SEED,
        'accuracy': float((cm.loc[cm['true_class'] == cm['predicted_class'], 'count'].sum() / cm['count'].sum()) if cm['count'].sum() else float('nan')),
        'macro_f1': float(per['f1_score'].mean()),
        'pm_precision': per_class_value('PrimaryMetabolites', 'precision'),
        'pm_recall': per_class_value('PrimaryMetabolites', 'recall'),
        'saccharides_recall': per_class_value('Saccharides', 'recall'),
        'lipids_recall': per_class_value('Lipids', 'recall'),
        'sac_to_pm': confusion('Saccharides', 'PrimaryMetabolites'),
        'lipids_to_pm': confusion('Lipids', 'PrimaryMetabolites'),
        'pair_activation_rate': float(log['pair_penalty_activation_rate'].mean()),
        'pair_gate_weight_mean': float(log['pair_penalty_mean_gate_weight'].mean()),
        'pm_prob_mean': float(pm_probs.mean()),
        'pm_prob_std': float(pm_probs.std(ddof=1)) if len(pm_probs) > 1 else 0.0,
    })

    target_classes = ['Saccharides', 'Lipids', 'PrimaryMetabolites']
    color_map = {
        'Saccharides': 'tab:blue',
        'Lipids': 'tab:orange',
        'PrimaryMetabolites': 'tab:green',
    }
    bins = np.linspace(0.0, 1.0, 21)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, class_name in zip(axes, target_classes):
        values = preds.loc[preds['true_class'] == class_name, 'p_PrimaryMetabolites'].to_numpy()
        ax.hist(values, bins=bins, density=True, alpha=0.8, color=color_map[class_name], edgecolor='white')
        ax.axvline(tau, color='black', linestyle='--', linewidth=1.2)
        ax.set_title(f'True {class_name} (n={len(values)})')
        ax.set_xlabel('p_PM')
        ax.set_xlim(0.0, 1.0)
    axes[0].set_ylabel('Density')
    fig.suptitle(f'p_PM distributions by true class, tau={tau:.3f}', fontsize=13)
    fig.tight_layout()
    fig_path = HIST_DIR / f'{run_prefix}_pm_hist.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {fig_path}')

    print(
        f"tau {tau:.3f}: acc={summary_rows[-1]['accuracy']:.4f} macro_f1={summary_rows[-1]['macro_f1']:.4f} "
        f"pm_p={summary_rows[-1]['pm_precision']:.4f} pm_r={summary_rows[-1]['pm_recall']:.4f} "
        f"sac->pm={summary_rows[-1]['sac_to_pm']} lip->pm={summary_rows[-1]['lipids_to_pm']} "
        f"act={summary_rows[-1]['pair_activation_rate']:.4f}",
        flush=True,
    )

summary_df = pd.DataFrame(summary_rows)
hist_df = pd.DataFrame(hist_rows)
summary_df.to_csv(SUMMARY_DIR / 'threshold_sweep_summary.csv', index=False)
hist_df.to_csv(SUMMARY_DIR / 'pm_probability_histogram_summary.csv', index=False)

print('\n=== Threshold Sweep Summary ===')
for _, row in summary_df.iterrows():
    print(
        f"tau={row['tau']:.3f} acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
        f"pm_p={row['pm_precision']:.4f} pm_r={row['pm_recall']:.4f} "
        f"sac->pm={int(row['sac_to_pm'])} lip->pm={int(row['lipids_to_pm'])} "
        f"act={row['pair_activation_rate']:.4f} gate_mean={row['pair_gate_weight_mean']:.6f}"
    )
