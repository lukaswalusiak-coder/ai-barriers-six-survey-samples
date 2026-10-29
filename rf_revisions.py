"""Repeat RF fitting across ten fixed model seeds and two sensitivities.

Usage: python rf_revisions.py all|seeds|sensitivities|summarize
Requires private/analysis_data.csv and the original primary analysis outputs.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/revision/rf'
PRIVATE = ROOT / 'private/rf_revision'
OUT.mkdir(exist_ok=True, parents=True)
ORIGINAL_SEED = 20261014
SEEDS = list(range(20261014, 20261024))

def run(label, seed, data, regime='loso', mode='full'):
    root = OUT / label
    root.mkdir(exist_ok=True)
    a.OUT = root / 'results'
    a.OUT.mkdir(exist_ok=True)
    a.PRIV = PRIVATE / label
    a.PRIV.mkdir(exist_ok=True, parents=True)
    a.MODEL_SEED = seed
    a.SPLIT_SEED = ORIGINAL_SEED
    started = time.monotonic()
    a.run(data, regime, ['rf'], mode)
    meta = {'seed': seed, 'regime': regime, 'mode': mode,
            'n': len(data), 'elapsed_seconds': time.monotonic() - started}
    (root / 'run.json').write_text(json.dumps(meta, indent=2))

def summarize():
    seed_rows, configurations, checks, runs = [], [], [], []
    original = pd.read_csv(ROOT / 'results/metrics_loso_full.csv').set_index(['model', 'site'])
    lr_loss = float(original.loc[('lr', 'Equal-site mean'), 'log_loss'])
    for seed in SEEDS:
        result_dir = ROOT / 'results' if seed == ORIGINAL_SEED else OUT / f'seed_{seed}/results'
        private_dir = ROOT / 'private' if seed == ORIGINAL_SEED else PRIVATE / f'seed_{seed}'
        metrics = pd.read_csv(result_dir / 'metrics_loso_full.csv').set_index(['model', 'site'])
        row = metrics.loc[('rf', 'Equal-site mean')].to_dict()
        row.update({'seed': seed, 'prior_log_loss': float(metrics.loc[('prior', 'Equal-site mean'), 'log_loss'])})
        row['gain_over_prior'] = row['prior_log_loss'] - row['log_loss']
        row['lr_minus_rf_log_loss'] = lr_loss - row['log_loss']
        seed_rows.append(row)
        configs = json.loads((result_dir / 'config_loso_full.json').read_text())
        rf_configs = [c for c in configs if c['family'] == 'rf']
        assert len(rf_configs) == 6
        configurations.extend({'seed': seed, **c} for c in rf_configs)
        # The archived aggregate outputs can be summarized without private predictions.
        # If rerunning from records, also verify every available prediction matrix.
        prediction_file = private_dir / 'predictions_loso_full.npz'
        if prediction_file.exists():
            with np.load(prediction_file) as arrays:
                for name in arrays.files:
                    p = arrays[name]
                    assert p.shape == (3528, 10) and np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
            valid = True
        else:
            valid = None
        checks.append({'seed': seed, 'prediction_arrays_valid': valid, 'number_outer_folds': 6})
        if seed != ORIGINAL_SEED:
            runs.append(json.loads((OUT / f'seed_{seed}/run.json').read_text()))
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(OUT / 'rf_seed_metrics.csv', index=False)
    configs = pd.DataFrame(configurations)
    configs.drop(columns=['candidates']).to_csv(OUT / 'rf_seed_configs.csv', index=False)
    statistics = {k: {'mean': float(seeds[k].mean()), 'sd': float(seeds[k].std(ddof=1)),
                      'min': float(seeds[k].min()), 'max': float(seeds[k].max())}
                  for k in ['log_loss', 'accuracy', 'macro_f1', 'gain_over_prior', 'lr_minus_rf_log_loss']}
    parameters = {str(site): {str(int(r.seed)): int(r.parameter) for _, r in group.iterrows()}
                  for site, group in configs.groupby('fold')}
    sensitivity = {}
    for label, regime, mode in [('quality', 'quality', 'full'), ('no_concern', 'loso', 'no_concern')]:
        result_dir = OUT / label / 'results'
        metrics = pd.read_csv(result_dir / f'metrics_{regime}_{mode}.csv').set_index(['model', 'site'])
        sensitivity[label] = {m: metrics.loc[(m, 'Equal-site mean')].to_dict() for m in ['prior', 'rf', 'rf_temp']}
        for model in ['rf', 'rf_temp']:
            sensitivity[label][f'{model}_gain_over_prior'] = sensitivity[label]['prior']['log_loss'] - sensitivity[label][model]['log_loss']
        runs.append(json.loads((OUT / label / 'run.json').read_text()))
    report = {'protocol': 'Ten model seeds 20261014-20261023; original fixed LOSO partitions; full tuning and temperature calibration rerun for every model seed. No probability averaging or seed selection. Seed variation does not quantify sampling uncertainty. Sensitivities use 20261014.',
              'baseline_rf_loso': seed_rows[0], 'seed_count': len(SEEDS),
              'seed_statistics': statistics, 'parameters_by_site_and_seed': parameters,
              'rf_sensitivities': sensitivity, 'runs': runs, 'validation_checks': checks}
    (OUT / 'verified_report.json').write_text(json.dumps(report, indent=2))
    print(seeds[['seed', 'log_loss', 'gain_over_prior']].to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['all', 'seeds', 'sensitivities', 'summarize'])
    task = parser.parse_args().task
    if task != 'summarize':
        import analyze as a
    with threadpool_limits(limits=1):
        if task in ['all', 'seeds']:
            for seed in SEEDS[1:]:
                run(f'seed_{seed}', seed, a.D)
        if task in ['all', 'sensitivities']:
            run('quality', ORIGINAL_SEED, a.D.loc[~a.D.quality_flag].reset_index(drop=True), 'quality')
            run('no_concern', ORIGINAL_SEED, a.D, 'loso', 'no_concern')
        if task in ['all', 'summarize']:
            summarize()
