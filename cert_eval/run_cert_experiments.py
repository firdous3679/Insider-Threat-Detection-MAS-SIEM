from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from cert_eval.cert_loader import load_cert_data
from cert_eval.cert_schema import build_normalized_events
from cert_eval.cert_feature_builder import build_user_day_features
from cert_eval.cert_label_builder import build_labels
from cert_eval.cert_baselines import run_baselines
from cert_eval.cert_eg_siem_runner import run_cert_eg_siem
from cert_eval.cert_scalability import run_scalability
from cert_eval.cert_metrics import compute_classification_metrics, summarize_statistics, create_plots


def _evaluate_method(df, method, pred_col='pred_alert', score_col='risk_score'):
    y=df['user_day_label'].astype(int)
    pred=df[pred_col].astype(int)
    score=df[score_col] if score_col in df else pred
    met=compute_classification_metrics(y,pred,score)
    fp=((pred==1)&(y==0)).sum(); days=max(1,df['day'].nunique())
    met.update({'method':method,'fp_per_day':fp/days,'ttd_hours':0.0})
    return met


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/cert_r5.2')
    ap.add_argument('--output_dir', default='results/cert_r52')
    ap.add_argument('--max_users', type=int, default=2000)
    ap.add_argument('--random_seed', type=int, default=42)
    args=ap.parse_args()

    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    bundle=load_cert_data(args.data_dir)
    events=build_normalized_events(bundle)
    events.to_csv(out/'cert_normalized_events.csv', index=False)

    feats=build_user_day_features(bundle, out)
    labeled=build_labels(feats, bundle.answers, out)
    labeled=labeled.sort_values(['user','day']).reset_index(drop=True)

    baseline_df=run_baselines(labeled, random_seed=args.random_seed)
    baseline_df.to_csv(out/'baseline_results.csv', index=False)

    modes=[('CERT-LSC','lsc'),('CERT-EG-SIEM without email','without_email'),('CERT-EG-SIEM email only','email_only'),('CERT-EG-SIEM without ToM-like evidence','without_tom'),('CERT-EG-SIEM full','full')]
    ablation=[]
    for mname,mode in modes:
        pred_df=run_cert_eg_siem(labeled, mode=mode)
        met=_evaluate_method(pred_df,mname)
        ablation.append(met)
    abl_df=pd.DataFrame(ablation)
    abl_df.to_csv(out/'cert_ablation_results.csv', index=False)

    scal_df=run_scalability(labeled, user_sizes=(100,500,1000,args.max_users))
    scal_df.to_csv(out/'scalability_results.csv', index=False)

    all_results=pd.concat([baseline_df, abl_df], ignore_index=True, sort=False)
    summary=[]
    for metric in ['precision','recall','f1','roc_auc','pr_auc','fp_per_day']:
        s=summarize_statistics(all_results[metric].fillna(0).tolist())
        s['metric']=metric; summary.append(s)
    pd.DataFrame(summary).to_csv(out/'cert_statistical_summary.csv', index=False)
    create_plots(all_results, scal_df, out)

    # Paper-ready tables
    pd.DataFrame([{
        'Dataset':'CERT r5.2','Users':labeled['user'].nunique(),'Duration':f"{labeled['day'].nunique()} days",'Modalities':'logon/device/file/http/email/LDAP','Insider scenarios':bundle.answers.get('scenario', pd.Series()).nunique() if not bundle.answers.empty else 0,'Evaluation unit':'user-day','Purpose in this paper':'External benchmark generalization'
    }]).to_csv(out/'table_a_dataset_summary.csv', index=False)
    all_results.to_csv(out/'table_b_external_benchmark_results.csv', index=False)
    scal_df.to_csv(out/'table_c_scalability.csv', index=False)
    pd.DataFrame([{'Comparison':'CERT-EG-SIEM full vs CERT-LSC','Metric':'F1','Mean difference':float((all_results[all_results.method=='CERT-EG-SIEM full']['f1'].mean() - all_results[all_results.method=='CERT-LSC']['f1'].mean()) if 'method' in all_results else 0),'Test used':'descriptive','p-value':None,'Significant?':'N/A'}]).to_csv(out/'table_d_statistical_significance.csv', index=False)


if __name__ == '__main__':
    main()
