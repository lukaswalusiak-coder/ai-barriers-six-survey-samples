"""Additional prior baselines and arithmetic contrasts; no model refitting.

Reads the reproduced private analysis data, the fixed original outer splits,
and published aggregate model metrics. Never exports record-level data.
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
import analyze as a

OUT = ROOT / 'results/revision/priors'
OUT.mkdir(exist_ok=True, parents=True)
ORIGINAL = ROOT / 'results'
D = a.D
Y = D.target.to_numpy()
SEED = 20261014
all_rows, gain_rows, fold_rows = [], [], []

for regime in ['loso', 'random']:
    if regime == 'loso':
        splits = [(np.flatnonzero(D.site != s), np.flatnonzero(D.site == s))
                  for s in sorted(D.site.unique())]
    else:
        splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(
            D, D.site.astype(str) + '_' + D.target.astype(str)))
    preds = {k: np.full((len(D), 10), np.nan) for k in
             ['respondent_weighted_add_one', 'equal_site_add_one', 'equal_site_unsmoothed']}
    for fold, (tr, te) in enumerate(splits, 1):
        assert not np.intersect1d(tr, te).size
        train = D.iloc[tr]
        counts = np.bincount(train.target, minlength=10)
        preds['respondent_weighted_add_one'][te] = (counts + 1) / (len(tr) + 10)
        counts_by_site = np.array([np.bincount(g.target, minlength=10)
                                  for _, g in train.groupby('site', sort=True)])
        n_site = counts_by_site.sum(axis=1, keepdims=True)
        preds['equal_site_add_one'][te] = ((counts_by_site + 1) / (n_site + 10)).mean(axis=0)
        preds['equal_site_unsmoothed'][te] = (counts_by_site / n_site).mean(axis=0)
        fold_rows.append({'regime': regime, 'fold': fold,
                          'n_train': len(tr), 'n_test': len(te),
                          'n_training_sites': len(counts_by_site)})
    orig = pd.read_csv(ORIGINAL / f'metrics_{regime}_full.csv')
    for name, p in preds.items():
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
        per_site = []
        for site in sorted(D.site.unique()):
            ix = D.site.eq(site).to_numpy()
            row = {'regime': regime, 'baseline': name, 'site': site,
                   **a.metrics(Y[ix], p[ix])}
            all_rows.append(row)
            per_site.append(row)
        metric_names = [k for k in a.metrics(Y, p) if k != 'n']
        eq = {'regime': regime, 'baseline': name, 'site': 'Equal-site mean',
              'n': len(D), **{k: float(np.mean([x[k] for x in per_site])) for k in metric_names}}
        all_rows.append(eq)
        all_rows.append({'regime': regime, 'baseline': name, 'site': 'Pooled', **a.metrics(Y, p)})
        for _, m in orig.loc[orig.site.eq('Equal-site mean') & ~orig.model.eq('prior')].iterrows():
            gain_rows.append({'regime': regime, 'baseline': name, 'model': m.model,
                              'baseline_log_loss': eq['log_loss'],
                              'model_log_loss': m.log_loss,
                              'log_loss_gain': eq['log_loss'] - m.log_loss,
                              'relative_gain_percent': 100 * (eq['log_loss'] - m.log_loss) / eq['log_loss']})
        if name == 'respondent_weighted_add_one':
            old = orig.loc[orig.site.eq('Equal-site mean') & orig.model.eq('prior')].iloc[0]
            assert abs(old.log_loss - eq['log_loss']) < 1e-12

pd.DataFrame(all_rows).to_csv(OUT / 'baseline_metrics.csv', index=False)
gains = pd.DataFrame(gain_rows)
gains.to_csv(OUT / 'model_gains_by_design.csv', index=False)
contrasts = []
for (baseline, model), g in gains.groupby(['baseline', 'model']):
    l = g.loc[g.regime.eq('loso')].iloc[0]
    r = g.loc[g.regime.eq('random')].iloc[0]
    raw = l.model_log_loss - r.model_log_loss
    prior_component = l.baseline_log_loss - r.baseline_log_loss
    gain_component = r.log_loss_gain - l.log_loss_gain
    assert abs(raw - prior_component - gain_component) < 1e-12
    contrasts.append({'baseline': baseline, 'model': model,
                      'model_loso_minus_random': raw,
                      'prior_loso_minus_random': prior_component,
                      'gain_random_minus_loso': gain_component})
pd.DataFrame(contrasts).to_csv(OUT / 'arithmetic_contrasts.csv', index=False)
pd.DataFrame(fold_rows).to_csv(OUT / 'outer_training_sizes.csv', index=False)
(OUT / 'analysis_notes.json').write_text(json.dumps({
    'status': 'computed_from_original_fixed_splits_and_aggregate_model_metrics',
    'equal_site_add_one': 'Average of (site_class_count+1)/(site_n+10) across available training sites.',
    'equal_site_unsmoothed': 'Smoothing sensitivity only; equal average of empirical training-site proportions.',
    'interpretation': 'Arithmetic contrast, not a causal decomposition of geographic or predictor-outcome shift.',
    'no_refitting': True, 'split_seed': SEED,
}, indent=2))
print(gains.loc[gains.model.eq('rf')].to_string(index=False))
