from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from .cert_metrics import compute_classification_metrics


def _prep_xy(df):
    feature_cols=[c for c in df.columns if c not in {'user','day','user_day_label','actor_label','scenario','role'}]
    x=df[feature_cols].fillna(0).astype(float)
    y=df['user_day_label'].astype(int).values
    return x,y

def _actor_metrics(df, pred_col):
    actor_true=df.groupby('user')['actor_label'].max()
    actor_pred=df.groupby('user')[pred_col].max()
    tp=((actor_true==1)&(actor_pred==1)).sum(); fp=((actor_true==0)&(actor_pred==1)).sum(); fn=((actor_true==1)&(actor_pred==0)).sum()
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f=2*p*r/(p+r) if p+r else 0
    return p,r,f

def run_baselines(labeled_df: pd.DataFrame, random_seed: int=42):
    x,y=_prep_xy(labeled_df)
    benign=labeled_df['user_day_label']==0
    x_train=x[benign]
    results=[]
    models={
        'Isolation Forest': IsolationForest(random_state=random_seed, contamination='auto'),
        'One-Class SVM': OneClassSVM(gamma='scale', nu=0.05),
    }
    for name,m in models.items():
        m.fit(x_train)
        score=-m.decision_function(x)
        pred=(m.predict(x)==-1).astype(int)
        met=compute_classification_metrics(y,pred,score)
        fp=((pred==1)&(y==0)).sum(); days=max(1,labeled_df['day'].nunique())
        ap,ar,af=_actor_metrics(labeled_df.assign(pred=pred),'pred')
        met.update({'method':name,'fp_per_day':fp/days,'actor_precision':ap,'actor_recall':ar,'actor_f1':af})
        results.append(met)
    return pd.DataFrame(results)
