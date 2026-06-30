import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SEEDS = [1, 2, 3, 4, 5]
SUMMARY_DIR = Path('outputs/comparisons/20260428_softgate_quadratic_sweep')
LOG_DIR = SUMMARY_DIR / 'seed_logs'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for seed in SEEDS:
    print(f'=== Seed {seed} ===', flush=True)
    cmd = [
        sys.executable,
        'train_cnn_raman.py',
        '--task', 'single',
        '--single-seed', str(seed),
        '--single-epochs', '150',
        '--single-batch-size', '64',
        '--single-aug-factor', '5',
        '--single-pair-penalty', 'Saccharides:PrimaryMetabolites:3',
        '--single-pair-penalty-lambda', '0.5',
        '--single-pair-penalty-start-epoch', '81',
        '--single-pair-penalty-ramp-epochs', '70',
        '--single-pair-penalty-confidence-threshold', '0.35',
        '--single-early-stop-patience', '1000',
        '--single-early-stop-min-epoch', '150',
        '--single-skip-baselines',
        '--single-skip-posthoc',
    ]
    log_path = LOG_DIR / f'seed_{seed}.log'
    with log_path.open('w', encoding='utf-8') as log_file:
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    per = pd.read_csv('outputs/logs/per_class_metrics.csv')
    cm = pd.read_csv('outputs/logs/confusion_matrix.csv')
    log = pd.read_csv('outputs/logs/training_log.csv')

    def recall_for(class_name: str) -> float:
        return float(per.loc[per['class'] == class_name, 'recall'].iloc[0])

    def confusion(true_name: str, pred_name: str) -> int:
        return int(cm.loc[
            (cm['true_class'] == true_name) & (cm['predicted_class'] == pred_name),
            'count'
        ].iloc[0])

    row = {
        'seed': seed,
        'accuracy': float(cm['count'].sum() and (cm.loc[cm['true_class'] == cm['predicted_class'], 'count'].sum() / cm['count'].sum())),
        'macro_f1': float(per['f1_score'].mean()),
        'lipids_recall': recall_for('Lipids'),
        'saccharides_recall': recall_for('Saccharides'),
        'sac_to_pm': confusion('Saccharides', 'PrimaryMetabolites'),
        'lipids_to_pm': confusion('Lipids', 'PrimaryMetabolites'),
        'pair_activation_rate': float(log['pair_penalty_activation_rate'].mean()),
        'pair_gate_weight_mean': float(log['pair_penalty_mean_gate_weight'].mean()),
    }
    rows.append(row)
    print(
        f"seed {seed}: acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
        f"lip_recall={row['lipids_recall']:.4f} sac_recall={row['saccharides_recall']:.4f} "
        f"sac->pm={row['sac_to_pm']} lip->pm={row['lipids_to_pm']} act={row['pair_activation_rate']:.4f}",
        flush=True,
    )

df = pd.DataFrame(rows)
df.to_csv(SUMMARY_DIR / 'seed_sweep_summary.csv', index=False)
print('\n=== Sweep Summary ===')
for metric in [
    'accuracy',
    'macro_f1',
    'lipids_recall',
    'saccharides_recall',
    'sac_to_pm',
    'lipids_to_pm',
    'pair_activation_rate',
    'pair_gate_weight_mean',
]:
    mean = float(df[metric].mean())
    std = float(df[metric].std(ddof=1)) if len(df) > 1 else 0.0
    print(f'{metric}: {mean:.4f} ± {std:.4f}')
