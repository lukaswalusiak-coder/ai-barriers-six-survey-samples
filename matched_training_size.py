"""Matched outer-training-size sensitivity; private observations never exported.

Uses original outer splits, nested site-blocked tuning, and temperature scaling.
Five training subsampling repetitions; model/split seeds remain fixed.
"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
from pathlib import Path
import importlib.util
import json
import time
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PRIV = ROOT / 'private/matched_size'
OUT = ROOT / 'results/revision/matched_size'
PRIV.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)
SOURCE = ROOT / 'analyze.py'
spec = importlib.util.spec_from_file_location('original_analysis', SOURCE)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
D = a.D.copy()
NTRAIN = 2670
SUBSAMPLE_SEEDS = [20261028, 20261029, 20261030, 20261031, 20261032]
MODEL_SEED = SPLIT_SEED = 20261014
FAMILIES = ['lr', 'rf']
a.MODEL_SEED = MODEL_SEED
a.SPLIT_SEED = SPLIT_SEED
protocol = {
    'n_outer_train': NTRAIN,
    'subsample_seeds': SUBSAMPLE_SEEDS,
    'model_seed': MODEL_SEED,
    'split_seed': SPLIT_SEED,
    'families': FAMILIES,
    'stratification': 'site x target; proportional allocation via StratifiedShuffleSplit',
    'evaluation': 'Original outer test folds retained. All 3528 respondents assessed per repetition.',
    'tuning': 'Original nested leave-one-training-site-out tuning and temperature calibration, refitted per subsample.',
    'prior_pooled': '(training category count + 1) / (training n + 10)',
    'prior_equal_site': 'Arithmetic mean across training sites of (site category count + 1) / (site n + 10)',
    'summary': 'Mean, sample SD, minimum and maximum of metrics over 5 subsampling repetitions, not averaged-prediction ensemble.',
    'limits': 'Matches outer training n only. Site composition, number of training sites, and inner training sizes still differ. Repetition range is not a confidence interval.',
    'analysis_source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
}
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
splits = {
    'loso': [(np.flatnonzero(D.site != s), np.flatnonzero(D.site == s)) for s in sorted(D.site.unique())],
    'random': list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED).split(D, D.site.astype(str) + '_' + D.target.astype(str))),
}
rows = []
configs = []
split_manifest = []
t0 = time.time()

def append_metrics(preds, repetition, subsample_seed, regime):
    y = D.target.to_numpy()
    for model, p in preds.items():
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
        site_metrics = []
        common = {'repetition': repetition, 'subsample_seed': subsample_seed, 'regime': regime, 'model': model}
        rows.append({**common, 'site': 'Pooled', **a.metrics(y, p)})
        for site in sorted(D.site.unique()):
            ix = D.site.eq(site).to_numpy()
            result = a.metrics(y[ix], p[ix])
            site_metrics.append(result)
            rows.append({**common, 'site': site, **result})
        rows.append({**common, 'site': 'Equal-site mean', 'n': len(D), **{k: float(np.mean([x[k] for x in site_metrics])) for k in site_metrics[0] if k != 'n'}})

with threadpool_limits(limits=1):
    for rep, subsample_seed in enumerate(SUBSAMPLE_SEEDS, 1):
        for regime_index, (regime, outer_splits) in enumerate(splits.items()):
            names = ['prior', 'prior_equal_site'] + FAMILIES + [f + '_temp' for f in FAMILIES]
            preds = {name: np.full((len(D), 10), np.nan) for name in names}
            coverage = np.zeros(len(D), dtype=int)
            for fold, (outer_train, test_idx) in enumerate(outer_splits):
                assert len(outer_train) >= NTRAIN
                draw_seed = int(np.random.SeedSequence([subsample_seed, regime_index, fold]).generate_state(1)[0])
                if len(outer_train) == NTRAIN:
                    selected = outer_train.copy()
                else:
                    candidate = D.iloc[outer_train]
                    labels = candidate.site.astype(str) + '_' + candidate.target.astype(str)
                    selection, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=NTRAIN, random_state=draw_seed).split(candidate, labels))
                    selected = np.sort(outer_train[selection])
                assert len(selected) == NTRAIN and len(np.unique(selected)) == NTRAIN
                assert np.intersect1d(selected, test_idx).size == 0
                assert set(selected).issubset(set(outer_train))
                coverage[test_idx] += 1
                train = D.iloc[selected].reset_index(drop=True)
                test = D.iloc[test_idx].reset_index(drop=True)
                expected_sites = 5 if regime == 'loso' else 6
                assert train.site.nunique() == expected_sites
                assert np.array_equal(np.sort(train.target.unique()), a.CLASSES)
                fold_name = str(test.site.iloc[0]) if regime == 'loso' else str(fold + 1)
                split_manifest.append({'repetition': rep, 'subsample_seed': subsample_seed, 'regime': regime, 'fold': fold_name, 'draw_seed': draw_seed, 'original_train_n': len(outer_train), 'selected_train_indices': selected.tolist(), 'test_indices': test_idx.tolist()})
                preds['prior'][test_idx] = (np.bincount(train.target, minlength=10) + 1) / (len(train) + 10)
                per_site = [(np.bincount(g.target, minlength=10) + 1) / (len(g) + 10) for _, g in train.groupby('site')]
                preds['prior_equal_site'][test_idx] = np.mean(per_site, axis=0)
                for family in FAMILIES:
                    p, pc, _, meta = a.fit_nested(train, test, family)
                    preds[family][test_idx] = p
                    preds[family + '_temp'][test_idx] = pc
                    configs.append({'repetition': rep, 'subsample_seed': subsample_seed, 'regime': regime, 'fold': fold_name, 'family': family, 'train_n': len(train), 'test_n': len(test), **meta})
                print(f'repetition {rep}/5 {regime} fold {fold_name} complete ({time.time()-t0:.1f}s)', flush=True)
            assert np.all(coverage == 1)
            append_metrics(preds, rep, subsample_seed, regime)
            np.savez_compressed(PRIV / f'predictions_{regime}_rep{rep}.npz', **preds)
            pd.DataFrame(rows).to_csv(OUT / 'metrics_matched_size.csv', index=False)
            (OUT / 'configs_matched_size.json').write_text(json.dumps(configs, indent=2))
            (PRIV / 'split_indices.json').write_text(json.dumps(split_manifest))

metrics_df = pd.DataFrame(rows)
mean_rows = metrics_df.loc[metrics_df.site.eq('Equal-site mean')]
summary_rows = []
for (regime, model), g in mean_rows.groupby(['regime', 'model']):
    result = {'regime': regime, 'model': model, 'n_repetitions': len(g)}
    for metric in ['log_loss', 'brier', 'accuracy', 'macro_f1', 'macro_recall', 'ece']:
        for stat, value in [('mean', g[metric].mean()), ('sd', g[metric].std(ddof=1)), ('min', g[metric].min()), ('max', g[metric].max())]:
            result[metric + '_' + stat] = float(value)
    summary_rows.append(result)
pd.DataFrame(summary_rows).to_csv(OUT / 'summary_matched_size.csv', index=False)

gain_rows = []
for (rep, seed, regime, site), g in metrics_df.groupby(['repetition', 'subsample_seed', 'regime', 'site']):
    losses = g.set_index('model').log_loss
    for model in FAMILIES + [f + '_temp' for f in FAMILIES]:
        for baseline in ['prior', 'prior_equal_site']:
            gain_rows.append({'repetition': rep, 'subsample_seed': seed, 'regime': regime, 'site': site, 'model': model, 'baseline': baseline, 'baseline_log_loss': float(losses[baseline]), 'model_log_loss': float(losses[model]), 'log_loss_gain': float(losses[baseline] - losses[model])})
gain_df = pd.DataFrame(gain_rows)
gain_df.to_csv(OUT / 'gains_matched_size.csv', index=False)

contrasts = []
for (rep, model, baseline), g in gain_df.loc[gain_df.site.eq('Equal-site mean')].groupby(['repetition', 'model', 'baseline']):
    g = g.set_index('regime')
    total = float(g.loc['loso', 'model_log_loss'] - g.loc['random', 'model_log_loss'])
    base = float(g.loc['loso', 'baseline_log_loss'] - g.loc['random', 'baseline_log_loss'])
    gain = float(g.loc['random', 'log_loss_gain'] - g.loc['loso', 'log_loss_gain'])
    assert np.isclose(total, base + gain, atol=1e-12)
    contrasts.append({'repetition': rep, 'model': model, 'baseline': baseline, 'model_loss_difference_loso_minus_random': total, 'baseline_loss_difference_loso_minus_random': base, 'gain_difference_random_minus_loso': gain})
pd.DataFrame(contrasts).to_csv(OUT / 'contrasts_matched_size.csv', index=False)
result = {'runtime_seconds': time.time() - t0, 'n_respondents': len(D), 'n_outer_train': NTRAIN, 'repetitions': len(SUBSAMPLE_SEEDS), 'n_outer_fits_per_family': len(configs) // len(FAMILIES), 'all_splits_disjoint': True, 'all_outer_test_coverage_once_per_regime_repetition': True, 'families': FAMILIES, 'summary': summary_rows}
(OUT / 'verification_summary.json').write_text(json.dumps(result, indent=2))
print('COMPLETE', json.dumps(result), flush=True)
