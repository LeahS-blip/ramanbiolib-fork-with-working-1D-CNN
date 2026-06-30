import subprocess
import sys
from pathlib import Path

import pandas as pd

SEEDS = [1, 2, 3]
MODES = [
    {
        'name': 'linear',
        'lambda': '0.5',
        'power': '1.0',
    },
    {
        'name': 'quadratic',
        'lambda': '0.5',
        'power': '2.0',
    },
]
SUMMARY_DIR = Path('outputs/comparisons/20260429_forbidden_mass_sweep')
LOG_DIR = SUMMARY_DIR / 'seed_logs'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for mode in MODES:
    print(f"=== Mode {mode['name']} ===", flush=True)
    for seed in SEEDS:
        print(f"--- Seed {seed} ---", flush=True)
        cmd = [
            sys.executable,
            'train_cnn_raman.py',
            '--task', 'single',
            '--single-seed', str(seed),
            '--single-epochs', '150',
            '--single-batch-size', '64',
            '--single-aug-factor', '5',
            '--single-forbidden-mass-lambda', mode['lambda'],
            '--single-forbidden-mass-power', mode['power'],
            '--single-early-stop-patience', '1000',
            '--single-early-stop-min-epoch', '150',
            '--single-skip-baselines',
            '--single-skip-posthoc',
        ]
        log_path = LOG_DIR / f"{mode['name']}_seed_{seed}.log"
        with log_path.open('w', encoding='utf-8') as log_file:
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)

        per = pd.read_csv('outputs/logs/per_class_metrics.csv')
        cm = pd.read_csv('outputs/logs/confusion_matrix.csv')
        log = pd.read_csv('outputs/logs/training_log.csv')
        cfg = pd.read_csv('outputs/logs/test_predictions.csv')

        def per_class_value(class_name: str, column: str) -> float:
            return float(per.loc[per['class'] == class_name, column].iloc[0])

        def confusion(true_name: str, pred_name: str) -> int:
            return int(cm.loc[
                (cm['true_class'] == true_name) & (cm['predicted_class'] == pred_name),
                'count'
            ].iloc[0])

        row = {
            'mode': mode['name'],
            'seed': seed,
            'accuracy': float((cm.loc[cm['true_class'] == cm['predicted_class'], 'count'].sum() / cm['count'].sum()) if cm['count'].sum() else float('nan')),
            'macro_f1': float(per['f1_score'].mean()),
            'pm_precision': per_class_value('PrimaryMetabolites', 'precision'),
            'pm_recall': per_class_value('PrimaryMetabolites', 'recall'),
            'saccharides_recall': per_class_value('Saccharides', 'recall'),
            'lipids_recall': per_class_value('Lipids', 'recall'),
            'sac_to_pm': confusion('Saccharides', 'PrimaryMetabolites'),
            'lipids_to_pm': confusion('Lipids', 'PrimaryMetabolites'),
            'train_forbidden_prob': float(log['forbidden_mass_mean_prob'].mean()) if 'forbidden_mass_mean_prob' in log else 0.0,
            'train_forbidden_term': float(log['forbidden_mass_mean_term'].mean()) if 'forbidden_mass_mean_term' in log else 0.0,
            'pm_prob_mean': float(cfg['p_PrimaryMetabolites'].mean()),
            'pm_prob_std': float(cfg['p_PrimaryMetabolites'].std(ddof=1)) if len(cfg) > 1 else 0.0,
        }
        rows.append(row)
        print(
            f"mode {mode['name']} seed {seed}: acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
            f"pm_p={row['pm_precision']:.4f} pm_r={row['pm_recall']:.4f} "
            f"sac->pm={row['sac_to_pm']} lip->pm={row['lipids_to_pm']}",
            flush=True,
        )

summary_df = pd.DataFrame(rows)
summary_df.to_csv(SUMMARY_DIR / 'forbidden_mass_sweep_summary.csv', index=False)

print('\n=== Forbidden-Mass Sweep Summary ===')
for mode_name, mode_df in summary_df.groupby('mode'):
    print(f'-- {mode_name} --')
    for metric in [
        'accuracy',
        'macro_f1',
        'pm_precision',
        'pm_recall',
        'saccharides_recall',
        'lipids_recall',
        'sac_to_pm',
        'lipids_to_pm',
        'train_forbidden_prob',
        'train_forbidden_term',
    ]:
        mean = float(mode_df[metric].mean())
        std = float(mode_df[metric].std(ddof=1)) if len(mode_df) > 1 else 0.0
        print(f'{metric}: {mean:.4f} ± {std:.4f}')
