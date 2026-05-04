from __future__ import annotations
import time, tracemalloc
import pandas as pd
from .cert_eg_siem_runner import run_cert_eg_siem


def run_scalability(labeled_df: pd.DataFrame, user_sizes=(100,500,1000,2000)) -> pd.DataFrame:
    rows=[]
    users=list(labeled_df['user'].astype(str).unique())
    for n in user_sizes:
        subset_users=set(users[:min(n,len(users))])
        sub=labeled_df[labeled_df['user'].astype(str).isin(subset_users)].copy()
        tracemalloc.start(); t0=time.perf_counter()
        out=run_cert_eg_siem(sub, mode='full')
        runtime=time.perf_counter()-t0
        _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
        events=len(out)
        rows.append({'num_users':len(subset_users),'events_processed':events,'runtime_seconds':runtime,'peak_memory_mb':peak/(1024*1024),'events_per_second':events/runtime if runtime>0 else 0})
    return pd.DataFrame(rows)
