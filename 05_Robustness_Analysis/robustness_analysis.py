from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'telco_clean.csv'
MAIN=ROOT/'results'/'main'
RED=ROOT/'results'/'without_total_charges'
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)

def auc(y,score):
    rank=pd.Series(score).rank(method='average').to_numpy()
    n_pos=int((y==1).sum()); n_neg=int((y==0).sum())
    return (rank[y==1].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg)

def f1(y,pred):
    tp=int(((y==1)&(pred==1)).sum()); fp=int(((y==0)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum())
    return 2*tp/max(1,2*tp+fp+fn)

# Confirm data profile and the paired row-level test predictions.
df=pd.read_csv(DATA)
a=pd.read_csv(MAIN/'rf_test_predictions.csv').sort_values('csv_row_0based').reset_index(drop=True)
b=pd.read_csv(RED/'rf_test_predictions.csv').sort_values('csv_row_0based').reset_index(drop=True)
assert df.shape==(7043,21)
assert a.csv_row_0based.equals(b.csv_row_0based), 'The two test ID lists differ.'
assert a.churn.equals(b.churn), 'The two test labels differ.'
assert len(a)==1409 and int(a.churn.sum())==374
weighted_same=bool(np.array_equal(a.unweighted_prediction,a.balanced_prediction))

# Calculate paired differences on the same customers.
y=a.churn.to_numpy(dtype=int)
p_full=a.unweighted_probability.to_numpy(); p_drop=b.unweighted_probability.to_numpy()
c_full=a.unweighted_prediction.to_numpy(dtype=int); c_drop=b.unweighted_prediction.to_numpy(dtype=int)
delta_auc=auc(y,p_drop)-auc(y,p_full)
delta_f1=f1(y,c_drop)-f1(y,c_full)
rng=np.random.default_rng(42); positive=np.flatnonzero(y==1); negative=np.flatnonzero(y==0); boot=np.empty((2000,2))
for i in range(2000):
    ix=np.r_[rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)]
    boot[i,0]=auc(y[ix],p_drop[ix])-auc(y[ix],p_full[ix])
    boot[i,1]=f1(y[ix],c_drop[ix])-f1(y[ix],c_full[ix])
ci=np.quantile(boot,[.025,.975],axis=0)
pd.DataFrame([
 {'comparison':'Without Total Charges minus full model','metric':'AUC','difference':delta_auc,'ci_low':ci[0,0],'ci_high':ci[1,0]},
 {'comparison':'Without Total Charges minus full model','metric':'F1','difference':delta_f1,'ci_low':ci[0,1],'ci_high':ci[1,1]},
]).to_csv(OUT/'paired_robustness_differences.csv',index=False,float_format='%.8f')

metrics=pd.concat([
 pd.read_csv(MAIN/'rf_test_metrics.csv').assign(specification='With Total Charges'),
 pd.read_csv(RED/'rf_test_metrics.csv').assign(specification='Without Total Charges')
],ignore_index=True)
metrics.to_csv(OUT/'rf_metrics_combined.csv',index=False)
sha=hashlib.sha256(DATA.read_bytes()).hexdigest()
audit=(f'Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns\n'
       f'Dataset SHA-256: {sha}\n'
       f'Paired test rows: {len(a)}; churn cases: {int(a.churn.sum())}\n'
       f'Main and no-Total-Charges row IDs and labels align: yes\n'
       f'Weighted/unweighted hard predictions identical in supplied results: {weighted_same}\n'
       f'Delta AUC (without - with): {delta_auc:.8f}; paired bootstrap 95% CI [{ci[0,0]:.8f}, {ci[1,0]:.8f}]\n'
       f'Delta F1 (without - with): {delta_f1:.8f}; paired bootstrap 95% CI [{ci[0,1]:.8f}, {ci[1,1]:.8f}]\n')
(OUT/'audit_summary.txt').write_text(audit,encoding='utf-8')
print(audit)
