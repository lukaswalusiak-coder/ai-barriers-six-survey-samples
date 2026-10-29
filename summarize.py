"""Rebuild aggregate results from private records and out-of-fold predictions.
Run after all three analyze.py commands; plot_aggregate_results.py needs no private data.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

R=Path(__file__).resolve().parent; O=R/'results'; F=R/'figures'; F.mkdir(exist_ok=True)
d=pd.read_csv(R/'private/analysis_data.csv'); y=d.target.to_numpy(); sites=sorted(d.site.unique())
p=dict(np.load(R/'private/predictions_loso_full.npz')); q=dict(np.load(R/'private/predictions_random_full.npz'))
labels=['Trust','Knowledge / skills','Privacy / data','Jobs / unfair decisions','Errors / quality','Rules / accountability','Cost','Device / internet access','No need','Hard to say']
names={'prior':'Training prior','demo_lr':'Demographics + use','lr':'Full logistic regression','rf':'Random forest','hgb':'Gradient boosting','demo_lr_temp':'Demographics + use (T)','lr_temp':'Full logistic regression (T)','rf_temp':'Random forest (T)','hgb_temp':'Gradient boosting (T)'}
for arr in [*p.values(),*q.values()]:
    assert arr.shape==(3528,10) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1)
loss={k:-np.log(v[np.arange(len(y)),y]) for k,v in p.items()}
rloss={k:-np.log(v[np.arange(len(y)),y]) for k,v in q.items()}
ixs=[np.flatnonzero(d.site.eq(s)) for s in sites]
def equal(x):return float(np.mean([x[ix].mean() for ix in ixs]))
contrasts={'LR minus prior':loss['lr']-loss['prior'],'RF minus prior':loss['rf']-loss['prior'],'HGB T minus prior':loss['hgb_temp']-loss['prior'],'LR minus demographic LR':loss['lr']-loss['demo_lr'],'RF minus LR':loss['rf']-loss['lr'],'RF T minus RF':loss['rf_temp']-loss['rf'],'LOSO minus random RF':loss['rf']-rloss['rf'],'LOSO minus random LR':loss['lr']-rloss['lr']}
rng=np.random.default_rng(20261014); boot={k:[] for k in contrasts}
for b in range(2000):
    sampled=[rng.choice(ix,len(ix),replace=True) for ix in ixs]
    for k,v in contrasts.items():boot[k].append(float(np.mean([v[ix].mean() for ix in sampled])))
ci=pd.DataFrame([{'contrast':k,'estimate':equal(v),'lower':np.quantile(boot[k],.025),'upper':np.quantile(boot[k],.975),'B':2000} for k,v in contrasts.items()])
ci.to_csv(O/'paired_bootstrap_logloss.csv',index=False)
counts=pd.crosstab(d.site,d.target).reindex(sites); counts.columns=labels
counts.to_csv(O/'barrier_counts.csv'); pct=counts.div(counts.sum(1),axis=0)*100; pct.to_csv(O/'barrier_percentages.csv')
desc=[]
for site,g in [('Pooled',d)]+[(s,d.loc[d.site.eq(s)]) for s in sites]:
    desc.append({'site':site,'n':len(g),'age_mean':g.age_years.mean(),'age_sd':g.age_years.std(),'age_median':g.age_years.median(),'female_n':int(g.sex.eq(1).sum()),'higher_education_n':int(g.education.isin([5,6,7]).sum()),'never_AI_n':int(g.p15.eq(1).sum()),'quality_excluded_n':int(g.quality_flag.sum())})
pd.DataFrame(desc).to_csv(O/'sample_description.csv',index=False)
rows=[]
for model,pr in p.items():
    for site in ['Pooled']+sites:
        ix=np.arange(len(d)) if site=='Pooled' else np.flatnonzero(d.site.eq(site))
        for k,label in enumerate(labels):
            yy=y[ix]; pred=pr[ix].argmax(1); actual=yy==k
            rows.append({'model':model,'site':site,'class':k,'label':label,'n':int(actual.sum()),'observed':float(actual.mean()),'mean_predicted':float(pr[ix,k].mean()),'predicted_minus_observed':float(pr[ix,k].mean()-actual.mean()),'recall':float((pred[actual]==k).mean())})
pd.DataFrame(rows).to_csv(O/'classwise_performance.csv',index=False)
# Pairwise outcome distribution distances use natural logs (0 to sqrt(log(2))).
js=np.array([[jensenshannon(counts.loc[a],counts.loc[b]) for b in sites] for a in sites])
pd.DataFrame(js,index=sites,columns=sites).to_csv(O/'outcome_js_distance.csv')
# Descriptive subgroup performance. No claims of equalized odds or causal fairness.
rows=[]
groups={'Age 18-34':d.age_years<35,'Age 35-54':d.age_years.between(35,54),'Age 55+':d.age_years>=55,'Women':d.sex.eq(1),'Men':d.sex.eq(2),'Other/prefer not sex':~d.sex.isin([1,2]),'Never uses AI':d.p15.eq(1),'Uses AI':d.p15>1}
for group,mask in groups.items():
    ix=np.flatnonzero(mask); pred=p['rf'][ix].argmax(1)
    rows.append({'group':group,'n':len(ix),'log_loss':float(loss['rf'][ix].mean()),'accuracy':float((pred==y[ix]).mean()),'small_n':bool(len(ix)<50)})
pd.DataFrame(rows).to_csv(O/'subgroups_rf.csv',index=False)
m=pd.read_csv(O/'metrics_loso_full.csv'); mr=pd.read_csv(O/'metrics_random_full.csv')
em=m[m.site.eq('Equal-site mean')].set_index('model'); er=mr[mr.site.eq('Equal-site mean')].set_index('model')
cw=pd.read_csv(O/'classwise_performance.csv')
imp=pd.read_csv(O/'permutation_importance.csv'); agg=imp.groupby('block').mean(numeric_only=True).sort_values('mean_loss_increase')
summary={'primary':em.reset_index().to_dict('records'),'random':er.reset_index().to_dict('records'),'bootstrap':ci.to_dict('records'),'sample':desc,'outcome_totals':counts.sum().to_dict(),'importance':agg.reset_index().to_dict('records'),'quality_n':int((~d.quality_flag).sum()),'source_n':len(d),'primary_rf_class_recall':cw[(cw.model=='rf')&cw.site.eq('Pooled')][['label','n','recall']].to_dict('records')}
for name in ['quality_full','loso_no_concern','loso_categorical']:
    z=pd.read_csv(O/f'metrics_{name}.csv');summary[name]=z[z.site.eq('Equal-site mean')].to_dict('records')
(O/'summary.json').write_text(json.dumps(summary,indent=2))
print(ci.to_string(index=False));print(em[['log_loss','accuracy','macro_f1','ece']].to_string());print(er[['log_loss','accuracy']].to_string());print(agg.to_string());print('VALIDATED: finite normalized predictions, complete N, bootstrap paired by site.')
