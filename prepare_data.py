from pathlib import Path
import pandas as pd, numpy as np, json, hashlib

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'extracted'/'data'
SITES={'GZM':'GZM','Istambuł':'Istanbul','Montreal':'Montreal','Nowy_Jork':'New York','Pekin':'Beijing','Praga':'Prague'}
LABELS=['Lack of trust','Lack of knowledge or skills','Privacy and data concerns','Job loss or unfair decisions','Errors or poor response quality','Unclear rules and accountability','High cost','Limited device or internet access','No need to use AI','Hard to say']

def prepare(source=SOURCE, output=ROOT/'private'):
    output.mkdir(exist_ok=True,parents=True)
    frames=[];audit=[];codebook={}
    for suffix,site in SITES.items():
        f=next(source.glob('*_'+suffix+'.xlsx'))
        xl=pd.ExcelFile(f); a=pd.read_excel(f,0);b=pd.read_excel(f,1)
        assert len(a)==len(b)
        if site=='GZM':
            # Verified by exact Polish column labels and the paired labelled worksheet.
            mapping={'id':0,'duration_min':1,'sex':2,'consent':4,'age_band':5,'age_years':6,'education':21}
            mapping.update({f'emp{k}':22+k for k in range(1,8)})
            mapping.update({f'p{k}':29+k for k in range(1,14)})
            mapping.update({'p15':43,'p17':54,'p18':55,'p19':56,'p20':57,'p21':58,'p23':72})
            d=pd.DataFrame({key:a.iloc[:,j] for key,j in mapping.items()})
            d['duration_min']=d.duration_min/60
        else:
            a.columns=[str(c).lower() for c in a.columns]
            names={'id':'id','duration_min':'survey_finish_time','sex':'sex','consent':'zgoda','age_band':'age','age_years':'year','education':'wyksztalcenie'}
            names.update({f'emp{k}':f'status_zawodowy_r{k}' for k in range(1,8)})
            names.update({f'p{k}':f'p{k}' for k in list(range(1,14))+[15,17,18,19,20,21,23]})
            d=pd.DataFrame({key:a[col] for key,col in names.items()})
            mapping={key:a.columns.get_loc(col) for key,col in names.items()}
        for key,j in mapping.items():
            codebook.setdefault(key,{})[site]={'source_column':str(a.columns[j]),'labelled_column':str(b.columns[j]),'position_zero_based':j}
        d['site']=site
        assert d.consent.eq(1).all()
        assert d.age_years.between(18,100).all()
        assert d.p23.between(1,10).all()
        assert d[['p15','p18','p19','p20','p21']].isin(range(1,8)).all().all()
        assert d[[f'emp{k}' for k in range(1,8)]].isin([0,1]).all().all()
        assert d[['id','age_years','sex','education','p23']].notna().all().all()
        assert d.id.is_unique
        assert (d.p17.eq(99)==d.p15.eq(1)).all()
        d['quality_flag']=(d.duration_min<3)|(d[[f'p{k}' for k in range(1,14)]].nunique(axis=1)==1)
        d['target']=d.p23.astype(int)-1
        frames.append(d)
        audit.append({'site':site,'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'n':len(d),'columns':len(a.columns),'sheets':xl.sheet_names,'all_consent':True,'duplicate_ids':int(d.id.duplicated().sum()),'duration_median':float(d.duration_min.median()),'under3':int((d.duration_min<3).sum()),'straightline':int((d[[f'p{k}' for k in range(1,14)]].nunique(axis=1)==1).sum()),'quality_flag':int(d.quality_flag.sum()),'start_min':str(a['survey_start_time'].min()) if 'survey_start_time' in a else None,'start_max':str(a['survey_start_time'].max()) if 'survey_start_time' in a else None})
    data=pd.concat(frames,ignore_index=True)
    assert len(data)==3528
    assert not data.duplicated(['site','id']).any()
    data.to_csv(output/'analysis_data.csv',index=False)
    (output/'source_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    (output/'codebook.json').write_text(json.dumps(codebook,ensure_ascii=False,indent=2))
    print(pd.crosstab(data.site,data.target).to_string())
    print('n',len(data),'quality exclusions',int(data.quality_flag.sum()),'age',data.age_years.min(),data.age_years.max())
    return data

if __name__=='__main__': prepare()
