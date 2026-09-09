"""Demonstration baseline fitting without requiring the private raw dataset.

Run without arguments to generate a synthetic example. For real data, provide
a long-format CSV with curve_id,time,voltage,label and keep the file local.
"""
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def demo(n_curves=12, n_points=1000, seed=2026):
    rng=np.random.default_rng(seed); t=np.linspace(0,250,n_points); rows=[]
    for i in range(n_curves):
        drift=.01*np.sin(t/55)+.00003*t+rng.normal(0,.001,n_points)
        for ti,vi in zip(t,drift): rows.append((f"demo_{i+1:02d}",ti,vi,"No leakage"))
    return pd.DataFrame(rows,columns=["curve_id","time","voltage","label"])

def main():
    p=argparse.ArgumentParser(); p.add_argument("input",nargs="?",type=Path); p.add_argument("--outdir",type=Path,default=Path("baseline_example_results")); p.add_argument("--window",type=int,default=51); a=p.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    df=demo() if a.input is None else pd.read_csv(a.input,encoding="utf-8-sig")
    required={"curve_id","time","voltage","label"}
    if not required.issubset(df.columns): raise ValueError(f"Input requires columns: {sorted(required)}")
    df["time"]=pd.to_numeric(df.time,errors="coerce"); df["voltage"]=pd.to_numeric(df.voltage,errors="coerce"); df=df.dropna(subset=["curve_id","time","voltage"]); no=df[df.label.astype(str).str.lower().str.replace(" ","",regex=False).isin(["noleakage","normal","0","无泄漏","正常"])]
    if no.empty: raise ValueError("No no-leakage curves found")
    grid=np.linspace(no.time.min(),no.time.max(),1000); curves=[]
    for cid,g in no.groupby("curve_id"):
        g=g.sort_values("time"); curves.append(np.interp(grid,g.time,g.voltage))
    raw=np.nanmedian(np.vstack(curves),axis=0); w=min(a.window,len(grid)-1 if len(grid)%2==0 else len(grid)); w=max(5,w-(w%2==0)); baseline=savgol_filter(raw,w,3)
    pd.DataFrame({"time":grid,"baseline_raw":raw,"baseline":baseline}).to_csv(a.outdir/"demo_baseline.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,4),dpi=180); [ax.plot(grid,c,alpha=.2,color="gray") for c in curves]; ax.plot(grid,baseline,color="red",lw=2,label="fitted baseline"); ax.set(xlabel="Time",ylabel="Voltage"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(a.outdir/"demo_baseline.png",bbox_inches="tight"); plt.close(fig)
    print("Baseline demonstration complete:",a.outdir.resolve())
if __name__=="__main__": main()
