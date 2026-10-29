"""Nested evaluation of ten-category perceived AI barrier prediction.
Run after prepare_data.py. Record-level predictions stay in private/.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import json,time,platform,sys
import numpy as np,pandas as pd,scipy,sklearn
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss,f1_score,recall_score,accuracy_score,confusion_matrix
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
PRIV=ROOT/'private'; D=pd.read_csv(PRIV/'analysis_data.csv')
CLASSES=np.arange(10); SEED=20261014
DEMO=['age_years','sex','education']+[f'emp{k}' for k in range(1,8)]+['p15']
FULL=DEMO+['p18','p19','p20','p21']
GRID={'demo_lr':[.01,.1,1,10],'lr':[.01,.1,1,10],'rf':[10,30],'hgb':[7,15]}

def make_model(family,param,mode='full'):
    features=DEMO if family=='demo_lr' else FULL
    if mode=='no_concern':features=[c for c in features if c!='p21']
    cats=['sex','education']
    if mode=='categorical': cats+=['p15','p18','p19','p20','p21']
    nums=[c for c in features if c not in cats]
    prep=ColumnTransformer([('num',StandardScaler(),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
    if family in ['lr','demo_lr']: m=LogisticRegression(C=param,solver='lbfgs',max_iter=1500,random_state=SEED)
    elif family=='rf':m=RandomForestClassifier(n_estimators=200,min_samples_leaf=param,max_features='sqrt',n_jobs=1,random_state=SEED)
    else:m=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=param,min_samples_leaf=30,l2_regularization=10,early_stopping=False,random_state=SEED)
    return Pipeline([('preprocess',prep),('model',m)])

def site_weights(groups):
    g=np.asarray(groups);return np.array([1/np.sum(g==s) for s in g])

def fit_nested(train,test,family,mode='full'):
    groups=train.site.to_numpy(); y=train.target.to_numpy();candidates=[]
    for param in GRID[family]:
        oof=np.full((len(train),10),np.nan);loss=[]
        for s in sorted(set(groups)):
            va=groups==s;tr=~va;m=make_model(family,param,mode)
            m.fit(train.loc[tr],y[tr]);p=m.predict_proba(train.loc[va]);assert np.array_equal(m.classes_,CLASSES)
            oof[va]=p;loss.append(log_loss(y[va],p,labels=CLASSES))
        candidates.append((float(np.mean(loss)),param,oof))
    best=min(candidates,key=lambda a:a[0]);_,param,oof=best
    w=site_weights(groups); logp=np.log(np.clip(oof,1e-15,1))
    def objective(logt):
        p=softmax(logp/np.exp(logt),axis=1)
        return float(np.average(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)),weights=w))
    opt=minimize_scalar(objective,bounds=(np.log(.25),np.log(4)),method='bounded')
    T=float(np.exp(opt.x))
    m=make_model(family,param,mode);m.fit(train,y);p=m.predict_proba(test)
    pc=softmax(np.log(np.clip(p,1e-15,1))/T,axis=1)
    return p,pc,m,{'parameter':param,'temperature':T,'inner_loss':best[0],'candidates':[{'value':a[1],'loss':a[0]} for a in candidates]}

def ece(y,p):
    conf=p.max(axis=1);correct=(p.argmax(axis=1)==y).astype(float);v=0.
    bins=np.minimum((conf*10).astype(int),9)
    for k in range(10):
        ix=bins==k
        if ix.any():v+=ix.mean()*abs(correct[ix].mean()-conf[ix].mean())
    return float(v)

def metrics(y,p):
    pred=p.argmax(axis=1)
    return {'n':len(y),'log_loss':float(log_loss(y,p,labels=CLASSES)),'brier':float(np.mean(np.sum((p-np.eye(10)[y])**2,axis=1))),'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=CLASSES,average='macro',zero_division=0)),'macro_recall':float(recall_score(y,pred,labels=CLASSES,average='macro',zero_division=0)),'ece':ece(y,p)}

def run(data,regime,families,mode='full'):
    n=len(data);preds={k:np.full((n,10),np.nan) for k in ['prior']+families+[f+'_temp' for f in families]}
    configs=[];importances=[]
    if regime=='random':
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
        splits=list(cv.split(data,data.site.astype(str)+'_'+data.target.astype(str)))
    else:splits=[(np.flatnonzero(data.site!=s),np.flatnonzero(data.site==s)) for s in sorted(data.site.unique())]
    for fold,(tr,te) in enumerate(splits):
        train=data.iloc[tr].reset_index(drop=True);test=data.iloc[te].reset_index(drop=True)
        name=str(test.site.iloc[0]) if regime!='random' else str(fold+1)
        prior=(np.bincount(train.target,minlength=10)+1)/(len(train)+10)
        preds['prior'][te]=prior
        for family in families:
            p,pc,model,meta=fit_nested(train,test,family,mode)
            preds[family][te]=p;preds[family+'_temp'][te]=pc
            configs.append({'regime':regime,'fold':name,'family':family,'mode':mode,**meta})
            print(regime,mode,name,family,'LL',round(log_loss(test.target,p,labels=CLASSES),4),'T',round(meta['temperature'],3),flush=True)
            if regime=='loso' and mode=='full' and family=='lr':
                rng=np.random.default_rng(SEED+fold)
                blocks={'Demographics':['age_years','sex','education']+[f'emp{k}' for k in range(1,8)],'AI-use frequency':['p15'],'Anticipated need':['p18'],'Benefit-risk appraisal':['p19'],'Everyday-task trust':['p20'],'Societal concern':['p21']}
                base=log_loss(test.target,p,labels=CLASSES)
                for label,cols in blocks.items():
                    vals=[]
                    for rep in range(30):
                        shuffled=test.copy();idx=rng.permutation(len(test));shuffled[cols]=test.iloc[idx][cols].to_numpy()
                        vals.append(log_loss(test.target,model.predict_proba(shuffled),labels=CLASSES)-base)
                    importances.append({'site':name,'block':label,'mean_loss_increase':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1))})
        np.savez_compressed(PRIV/f'predictions_{regime}_{mode}.npz',**preds)
        (OUT/f'config_{regime}_{mode}.json').write_text(json.dumps(configs,indent=2))
    rows=[]
    for name,p in preds.items():
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)
        rows.append({'model':name,'site':'Pooled',**metrics(data.target.to_numpy(),p)})
        site_rows=[]
        for site in sorted(data.site.unique()):
            ix=data.site.eq(site).to_numpy();z={'model':name,'site':site,**metrics(data.target.to_numpy()[ix],p[ix])};rows.append(z);site_rows.append(z)
        rows.append({'model':name,'site':'Equal-site mean','n':n,**{k:float(np.mean([z[k] for z in site_rows])) for k in metrics(data.target.to_numpy(),p) if k!='n'}})
        cm=confusion_matrix(data.target,p.argmax(axis=1),labels=CLASSES)
        pd.DataFrame(cm).to_csv(OUT/f'confusion_{regime}_{mode}_{name}.csv',index_label='actual_class')
    pd.DataFrame(rows).to_csv(OUT/f'metrics_{regime}_{mode}.csv',index=False)
    if importances:pd.DataFrame(importances).to_csv(OUT/'permutation_importance.csv',index=False)
    return preds

if __name__=='__main__':
    task=sys.argv[1] if len(sys.argv)>1 else 'primary'
    with threadpool_limits(limits=1):
        if task=='primary':run(D,'loso',['demo_lr','lr','rf','hgb'])
        elif task=='random':run(D,'random',['demo_lr','lr','rf','hgb'])
        elif task=='sensitivity':
            run(D.loc[~D.quality_flag].reset_index(drop=True),'quality',['lr'])
            run(D,'loso',['lr'],'no_concern')
            run(D,'loso',['lr'],'categorical')
    (OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'seed':SEED},indent=2))
