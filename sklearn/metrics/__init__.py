from __future__ import annotations
import numpy as np

def accuracy_score(y_true, y_pred):
    y_true=np.array(y_true); y_pred=np.array(y_pred)
    return float((y_true==y_pred).mean())

def precision_score(y_true, y_pred, zero_division=0):
    y_true=np.array(y_true); y_pred=np.array(y_pred)
    tp=((y_true==1)&(y_pred==1)).sum(); fp=((y_true==0)&(y_pred==1)).sum()
    return float(tp/(tp+fp)) if (tp+fp) else float(zero_division)

def recall_score(y_true, y_pred, zero_division=0):
    y_true=np.array(y_true); y_pred=np.array(y_pred)
    tp=((y_true==1)&(y_pred==1)).sum(); fn=((y_true==1)&(y_pred==0)).sum()
    return float(tp/(tp+fn)) if (tp+fn) else float(zero_division)

def f1_score(y_true, y_pred, zero_division=0):
    p=precision_score(y_true,y_pred,zero_division=zero_division); r=recall_score(y_true,y_pred,zero_division=zero_division)
    return float(2*p*r/(p+r)) if (p+r) else float(zero_division)

def confusion_matrix(y_true, y_pred):
    y_true=np.array(y_true); y_pred=np.array(y_pred)
    tn=((y_true==0)&(y_pred==0)).sum(); fp=((y_true==0)&(y_pred==1)).sum(); fn=((y_true==1)&(y_pred==0)).sum(); tp=((y_true==1)&(y_pred==1)).sum()
    return np.array([[tn, fp],[fn,tp]])

def roc_auc_score(y_true, y_score):
    y_true=np.array(y_true); y_score=np.array(y_score)
    pos=y_score[y_true==1]; neg=y_score[y_true==0]
    if len(pos)==0 or len(neg)==0: return 0.0
    wins=0.0
    for p in pos:
        wins += (p>neg).sum() + 0.5*(p==neg).sum()
    return float(wins/(len(pos)*len(neg)))
