"""Recreate all revision figures from public aggregate CSV files only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
RESULTS = ROOT / 'results'
SITES = ['Beijing', 'GZM', 'Istanbul', 'Montreal', 'New York', 'Prague']
LABELS = ['Trust', 'Knowledge / skills', 'Privacy / data', 'Jobs / unfair decisions',
          'Errors / quality', 'Rules / accountability', 'Cost', 'Device / internet access',
          'No need', 'Hard to say']
SHORT = ['Trust', 'Knowledge /\nskills', 'Privacy /\ndata', 'Jobs /\nunfairness',
         'Errors /\nquality', 'Rules', 'Cost', 'Device /\ninternet', 'No need', 'Hard to\nsay']
TICK_LABELS = [s.replace('\n', ' ') for s in SHORT]
BLUE, ORANGE, GREEN = '#174A71', '#B45C24', '#4C7C54'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight', 'pdf.fonttype': 42})

def save(fig, stem):
    fig.savefig(OUT / f'{stem}.png', dpi=300, facecolor='white')
    fig.savefig(OUT / f'{stem}.pdf', facecolor='white')
    plt.close(fig)

pct = pd.read_csv(RESULTS / 'barrier_percentages.csv', index_col=0).reindex(SITES)
assert list(pct.columns) == LABELS
assert np.allclose(pct.sum(axis=1), 100)
fig, ax = plt.subplots(figsize=(8.4, 4.25))
im = ax.imshow(pct.to_numpy(), vmin=0, vmax=30, cmap='Blues', aspect='auto')
ax.set_xticks(range(10), TICK_LABELS, fontsize=9, rotation=35, ha='right')
ax.set_yticks(range(6), SITES)
for i in range(6):
    for j in range(10):
        ax.text(j, i, f'{pct.iloc[i,j]:.1f}', ha='center', va='center',
                color='white' if pct.iloc[i,j] > 19 else 'black', fontsize=9)
fig.colorbar(im, ax=ax, label='Respondents (%)', shrink=.85, pad=.025)
fig.tight_layout()
save(fig, 'Figure_1_barriers')

m = pd.read_csv(RESULTS / 'metrics_loso_full.csv')
mr = pd.read_csv(RESULTS / 'metrics_random_full.csv')
em = m.loc[m.site.eq('Equal-site mean')].set_index('model')
er = mr.loc[mr.site.eq('Equal-site mean')].set_index('model')
families = ['demo_lr', 'lr', 'rf', 'hgb']
names = ['Demographics + use LR', 'Full logistic regression', 'Random forest', 'Gradient boosting']
loso_gain = em.loc['prior', 'log_loss'] - em.loc[families, 'log_loss'].to_numpy()
random_gain = er.loc['prior', 'log_loss'] - er.loc[families, 'log_loss'].to_numpy()
fig, axes = plt.subplots(2, 1, figsize=(7.3, 6.5), gridspec_kw={'height_ratios': [1, 1.2]})
yy = np.arange(len(families))
for i in yy:
    axes[0].plot([loso_gain[i], random_gain[i]], [i, i], color='.65', linewidth=1.3)
axes[0].scatter(loso_gain, yy, color=BLUE, s=45, label='Leave one site out', zorder=3)
axes[0].scatter(random_gain, yy, color=ORANGE, marker='s', s=40,
                label='Random five-fold', zorder=3)
axes[0].set_yticks(yy, names)
axes[0].invert_yaxis()
axes[0].set_xlim(0, .115)
axes[0].set_xlabel('Log-loss gain over the corresponding training prior')
axes[0].set_title('(a) Predictive gain under each evaluation design', loc='left', fontsize=11)
axes[0].legend(fontsize=9, loc='upper right', frameon=False)
axes[0].grid(axis='x', alpha=.18)
rf = m.loc[m.model.eq('rf') & m.site.isin(SITES)].set_index('site')
pri = m.loc[m.model.eq('prior') & m.site.isin(SITES)].set_index('site')
gains = pri.loc[SITES, 'log_loss'] - rf.loc[SITES, 'log_loss']
axes[1].barh(SITES, gains, color=BLUE, height=.65)
axes[1].invert_yaxis()
axes[1].set_xlabel('Log-loss gain over the corresponding training prior')
axes[1].set_title('(b) Random forest by held-out site', loc='left', fontsize=11)
axes[1].grid(axis='x', alpha=.18)
axes[1].set_axisbelow(True)
fig.tight_layout(h_pad=1.7)
save(fig, 'Figure_2_validation')

cw = pd.read_csv(RESULTS / 'classwise_performance.csv')
mat = cw.loc[cw.model.eq('rf') & cw.site.isin(SITES)].pivot(
    index='site', columns='class', values='predicted_minus_observed').reindex(SITES) * 100
imp = pd.read_csv(RESULTS / 'permutation_importance.csv')
agg = imp.groupby('block').mean(numeric_only=True).sort_values('mean_loss_increase')
fig, axes = plt.subplots(2, 1, figsize=(8.3, 7.7), gridspec_kw={'height_ratios': [1.2, 1]})
im = axes[0].imshow(mat.to_numpy(), vmin=-15, vmax=15, cmap='RdBu_r', aspect='auto')
axes[0].set_yticks(range(6), SITES)
axes[0].set_xticks(range(10), TICK_LABELS, fontsize=9, rotation=35, ha='right')
axes[0].set_title('(a) Random forest: predicted minus observed category frequency',
                  fontsize=11, loc='left', pad=10)
fig.colorbar(im, ax=axes[0], label='Percentage points', shrink=.85, pad=.025)
axes[1].barh(agg.index, agg.mean_loss_increase, color=BLUE, height=.65)
axes[1].set_xlabel('Increase in held-out log loss after permutation')
axes[1].set_title('(b) Logistic-regression feature reliance', fontsize=11, loc='left')
axes[1].axvline(0, color='.4', linewidth=.8)
axes[1].grid(axis='x', alpha=.18)
axes[1].set_axisbelow(True)
fig.tight_layout(h_pad=2.3)
save(fig, 'Figure_3_diagnostics')

recall = cw.loc[cw.site.eq('Pooled') & cw.model.isin(['lr', 'rf', 'hgb'])].pivot(
    index='class', columns='model', values='recall').reindex(range(10)) * 100
assert recall.notna().all().all()
fig, ax = plt.subplots(figsize=(7.6, 5.1))
yy = np.arange(10)
for offset, (family, color, label) in zip([-.24, 0, .24],
        [('lr', ORANGE, 'Full logistic regression'), ('rf', BLUE, 'Random forest'),
         ('hgb', GREEN, 'Gradient boosting')]):
    ax.barh(yy + offset, recall[family], height=.22, color=color, label=label)
ax.set_yticks(yy, LABELS)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel('Pooled category-specific recall (%)')
ax.grid(axis='x', alpha=.18)
ax.set_axisbelow(True)
ax.legend(loc='lower right', fontsize=9, frameon=False)
fig.tight_layout()
save(fig, 'Figure_S1_class_recall')
print('Created four PNG/PDF figure pairs from public aggregate CSV files only.')
