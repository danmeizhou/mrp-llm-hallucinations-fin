"""Offline revision analysis. Run: python reanalyse.py --data pooled_feature_table.csv --out results
Requires numpy, pandas, scipy, scikit-learn, statsmodels. No API calls.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,sklearn,statsmodels.api as sm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,accuracy_score,precision_score,recall_score,f1_score
p=argparse.ArgumentParser();p.add_argument('--data',default='research-source/pooled_risk_outputs/pooled_feature_table.csv');p.add_argument('--out',default='analysis/results');args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(args.data);d['y']=d.label.isin(['hallucinated','contradictory']).astype(int)
assert len(d)==378 and not d.duplicated(['id','model']).any()
for c in ['question','evidence','ambiguity','reasoning_depth']:
 assert d.groupby('id')[c].nunique().max()==1
num=['question_length','readability_fk','evidence_length','retrieval_similarity','ambiguity','reasoning_depth'];qe=num+['numerical_content','topic_regulation','type_factual','source_official'];ids=['model_llama70b','model_gptoss20b']
models={'LR':lambda:LogisticRegression(C=1,max_iter=2000,random_state=42),'RF':lambda:RandomForestClassifier(n_estimators=500,min_samples_leaf=5,random_state=42,n_jobs=-1),'GB':lambda:GradientBoostingClassifier(n_estimators=100,learning_rate=.1,max_depth=3,random_state=42)}
def features(tr,te,cols):
 a=tr[cols].copy();b=te[cols].copy()
 if 'retrieval_similarity' in cols:
  u=tr.drop_duplicates('id');v=TfidfVectorizer(stop_words='english',max_features=5000).fit(pd.concat([u.question,u.evidence]))
  for frame,target in [(tr,a),(te,b)]:
   q=v.transform(frame.question);e=v.transform(frame.evidence);target['retrieval_similarity']=np.asarray(q.multiply(e).sum(axis=1)).ravel()
 nc=[c for c in num if c in cols]
 if nc:
  scale=StandardScaler();a[nc]=scale.fit_transform(a[nc]);b[nc]=scale.transform(b[nc])
 return a,b
splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(d,d.y,groups=d.id))
rows=[];preds=[]
def metrics(y,pr):
 pred=pr>=.5
 return {'auc':roc_auc_score(y,pr) if len(np.unique(y))==2 else np.nan,'ap':average_precision_score(y,pr),'brier':brier_score_loss(y,pr),'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0)}
def evaluate(frame,folds,cols,modelname,tag,train_model_exclude=None):
 probabilities=pd.Series(np.nan,index=frame.index);foldmap={}
 for f,(tri,tei) in enumerate(folds):
  tr=frame.iloc[tri];te=frame.iloc[tei]
  if train_model_exclude is not None:
   tr=tr[tr.model!=train_model_exclude];te=te[te.model==train_model_exclude]
  assert set(tr.id).isdisjoint(te.id)
  if te.empty:continue
  assert tr.y.nunique()==2
  a,b=features(tr,te,cols);m=models[modelname]();m.fit(a,tr.y);probabilities.loc[te.index]=np.round(m.predict_proba(b)[:,1],12)
  foldmap.update({int(i):f for i in te.index})
 ix=probabilities.dropna().index;y=frame.loc[ix,'y'].to_numpy();pr=probabilities.loc[ix].to_numpy();met=metrics(y,pr)
 # Conditional uncertainty: resample questions, retaining all associated OOF predictions.
 rng=np.random.default_rng(42);groups=frame.loc[ix,'id'].to_numpy();uniq=np.unique(groups);indices=[np.flatnonzero(groups==i) for i in uniq];boots=[]
 for _ in range(1000):
  sample=np.concatenate([indices[i] for i in rng.integers(0,len(uniq),len(uniq))])
  if len(np.unique(y[sample]))==2:boots.append(roc_auc_score(y[sample],pr[sample]))
 met.update(auc_lo=float(np.quantile(boots,.025)),auc_hi=float(np.quantile(boots,.975)),analysis=tag,classifier=modelname,n=len(ix),events=int(y.sum()),bootstrap_valid=len(boots))
 rows.append(met)
 for i,prob in probabilities.dropna().items():preds.append({'analysis':tag,'classifier':modelname,'row':i,'id':frame.loc[i,'id'],'model':frame.loc[i,'model'],'y':int(frame.loc[i,'y']),'probability':prob,'fold':foldmap[i]})
 print(tag,modelname,{k:round(v,3) for k,v in met.items() if isinstance(v,float)},flush=True)
for tag,cols in [('Full',qe+ids),('Question/evidence',qe),('Identity only',ids)]:
 for name in (models if tag!='Identity only' else ['LR']):evaluate(d,splits,cols,name,tag)
primary=d[d.model=='llama-3.1-8b-instant'].copy();fs=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(primary,primary.y,primary.id));evaluate(primary,fs,qe,'LR','Within Llama-3.1-8B')
for generator in d.model.unique():evaluate(d,splits,qe,'LR','Leave out '+generator,generator)
pd.DataFrame(rows).to_csv(out/'performance.csv',index=False);pd.DataFrame(preds).to_csv(out/'out_of_fold_predictions.csv',index=False)
pd.crosstab(d.model,d.label).to_csv(out/'label_counts.csv')
# Descriptive full-data coefficients, with question-cluster robust uncertainty.
a,_=features(d,d,qe+ids);fit=sm.Logit(d.y,sm.add_constant(a)).fit(disp=False,cov_type='cluster',cov_kwds={'groups':d.id})
ci=fit.conf_int();pd.DataFrame({'coef':fit.params,'OR':np.exp(fit.params),'low':np.exp(ci[0]),'high':np.exp(ci[1]),'p':fit.pvalues}).to_csv(out/'cluster_robust_odds_ratios.csv')
json.dump({'seed':42,'sklearn_version':sklearn.__version__,'n':len(d),'events':int(d.y.sum()),'source_commit':'90e4cba3eafce06de161d0e3aae816ffc042683c','pseudo_r2':fit.prsquared,'auc_summary':'pooled out-of-fold predictions','bootstrap':'1000 question-cluster resamples of fixed OOF predictions; no refitting'},open(out/'metadata.json','w'),indent=2)
