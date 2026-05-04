from __future__ import annotations
import pandas as pd


def run_cert_eg_siem(labeled_df: pd.DataFrame, mode: str='full') -> pd.DataFrame:
    df=labeled_df.copy()
    # Evidence extraction and trust-adaptive weighting.
    auth = (df['after_hours_logon_count'] + df['weekend_logon_count'] + (df['logon_count']>8).astype(int))
    device = df['device_connect_count'] + df['after_hours_device_count']
    file_e = df['file_access_count']*0.2 + df['file_copy_count'] + df['sensitive_file_count']
    web = df['suspicious_http_count'] + df['after_hours_http_count']
    email = df['external_email_count'] + df['attachment_email_count'] + (df['unique_recipient_count']>10).astype(int)
    peer = df['role_peer_deviation_score'].clip(lower=0)
    tom_like = ((df['logon_count']>0)&((df['file_copy_count']>0)| (df['suspicious_http_count']>0)| (df['external_email_count']>0))).astype(int)

    if mode == 'without_email': email *= 0
    if mode == 'email_only':
        auth*=0; device*=0; file_e*=0; web*=0; peer*=0; tom_like*=0
    if mode == 'without_tom': tom_like*=0
    if mode == 'lsc':
        score = (auth>0).astype(int)+(file_e>0).astype(int)+(web>0).astype(int)+(email>0).astype(int)
        df['risk_score']=score
        df['pred_alert']=(score>=2).astype(int)
        return df

    trust_factor = 1.0 / (1.0 + df['actor_label']*0.05)
    df['risk_score'] = trust_factor*(1.2*auth + 1.0*device + 1.4*file_e + 1.1*web + 1.2*email + 0.8*peer + 1.5*tom_like)
    # Evidence gate: require both score and diversity.
    categories=((auth>0).astype(int)+(device>0).astype(int)+(file_e>0).astype(int)+(web>0).astype(int)+(email>0).astype(int)+(peer>1).astype(int)+(tom_like>0).astype(int))
    df['pred_alert']=((df['risk_score']>=3.0)&(categories>=2)).astype(int)
    return df
