from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from scipy import stats
import matplotlib.pyplot as plt


def compute_classification_metrics(y_true, y_pred, y_score=None):
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    out = {"precision":p,"recall":r,"f1":f}
    if y_score is not None and len(set(y_true))>1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        out["roc_auc"] = np.nan; out["pr_auc"] = np.nan
    return out

def summarize_statistics(values):
    a=np.array(values,dtype=float)
    mean,std=float(np.nanmean(a)),float(np.nanstd(a,ddof=1)) if len(a)>1 else 0.0
    ci=1.96*std/np.sqrt(len(a)) if len(a)>1 else 0.0
    return {"mean":mean,"std":std,"ci95_low":mean-ci,"ci95_high":mean+ci}

def paired_significance(a,b,test='wilcoxon'):
    if test=='t-test': stat,p=stats.ttest_rel(a,b,nan_policy='omit')
    else: stat,p=stats.wilcoxon(a,b)
    return float(stat), float(p)

def create_plots(results_df: pd.DataFrame, scalability_df: pd.DataFrame, output_dir: str|Path):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    results_df.boxplot(column='f1', by='method'); plt.suptitle(''); plt.title('F1 distributions')
    plt.savefig(out/'fig_f1_boxplot.png', bbox_inches='tight'); plt.close()
    if 'ttd_hours' in results_df:
        results_df.boxplot(column='ttd_hours', by='method'); plt.suptitle(''); plt.title('TTD distributions')
        plt.savefig(out/'fig_ttd_boxplot.png', bbox_inches='tight'); plt.close()
    if not scalability_df.empty:
        plt.plot(scalability_df['num_users'], scalability_df['runtime_seconds'], marker='o')
        plt.xlabel('Users'); plt.ylabel('Runtime (s)'); plt.title('Scalability runtime')
        plt.savefig(out/'fig_scalability_runtime.png', bbox_inches='tight'); plt.close()
    fp = results_df.groupby('method')['fp_per_day'].mean().reset_index()
    plt.bar(fp['method'], fp['fp_per_day']); plt.xticks(rotation=45, ha='right'); plt.title('False positive comparison')
    plt.savefig(out/'fig_false_positive_comparison.png', bbox_inches='tight'); plt.close()
