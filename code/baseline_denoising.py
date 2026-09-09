
"""Demonstration baseline fitting without requiring the private raw dataset.

Run without arguments to generate a synthetic example. For real data, provide
either a long-format CSV with curve_id,time,voltage,label or a two-column
Excel/CSV file containing time and voltage for one no-leakage curve.
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

def read_input(path):
    """Read either the repository long format or a two-column sample curve."""
    try:
        df = pd.read_excel(path)
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {str(c).strip().lower(): c for c in df.columns}
    if {"curve_id", "time", "voltage", "label"}.issubset(cols):
        return df.rename(columns={cols["curve_id"]:"curve_id", cols["time"]:"time",
                                  cols["voltage"]:"voltage", cols["label"]:"label"})
    if len(df.columns) < 2:
        raise ValueError("Input must contain either curve_id/time/voltage/label or two time-voltage columns")
    # The supplied no-leakage-sample.xlsx is a two-column time-voltage table.
    out = df.iloc[:, :2].copy()
    out.columns = ["time", "voltage"]
    out["curve_id"] = "no_leakage_sample"
    out["label"] = "No leakage"
    return out[["curve_id", "time", "voltage", "label"]]

def main():
    p=argparse.ArgumentParser(); p.add_argument("input",nargs="?",type=Path); p.add_argument("--outdir",type=Path,default=Path("baseline_example_results")); p.add_argument("--window",type=int,default=51); a=p.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    df=demo() if a.input is None else read_input(a.input)
    required={"curve_id","time","voltage","label"}
    if not required.issubset(df.columns): raise ValueError(f"Input requires columns: {sorted(required)}")
    df["time"]=pd.to_numeric(df.time,errors="coerce"); df["voltage"]=pd.to_numeric(df.voltage,errors="coerce"); df=df.dropna(subset=["curve_id","time","voltage"]); no=df[df.label.astype(str).str.lower().str.replace(" ","",regex=False).isin(["noleakage","normal","0","无泄漏","正常"])]
    if no.empty: raise ValueError("No no-leakage curves found")
    grid=np.linspace(no.time.min(),no.time.max(),1000); curves=[]
    for cid,g in no.groupby("curve_id"):
        g=g.sort_values("time"); curves.append(np.interp(grid,g.time,g.voltage))
    raw=np.nanmedian(np.vstack(curves),axis=0); w=min(a.window,len(grid)-1 if len(grid)%2==0 else len(grid)); w=max(5,w-(w%2==0)); baseline=savgol_filter(raw,w,3)
    corrected = raw - baseline
    result = pd.DataFrame({"time":grid,"baseline_raw":raw,"baseline":baseline,
                           "baseline_corrected_voltage":corrected})
    result.to_csv(a.outdir/"baseline_denoised_origin.csv",index=False)
    result.to_csv(a.outdir/"demo_baseline.csv",index=False)
    fig,axes=plt.subplots(2,1,figsize=(10,6),dpi=180,sharex=True)
    [axes[0].plot(grid,c,alpha=.2,color="gray") for c in curves]
    axes[0].plot(grid,raw,color="#2166ac",lw=1.2,label="no-leakage signal")
    axes[0].plot(grid,baseline,color="#d73027",lw=2,label="fitted baseline")
    axes[0].set_ylabel("Voltage"); axes[0].legend(frameon=False)
    axes[1].plot(grid,corrected,color="#1a9850",lw=1.2,label="baseline-corrected signal")
    axes[1].axhline(0,color="black",lw=.7); axes[1].set_xlabel("Time"); axes[1].set_ylabel("Corrected voltage")
    axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(a.outdir/"baseline_denoised.png",bbox_inches="tight"); plt.close(fig)
    # Keep the historical filename as a compatible copy for existing workflows.
    import shutil
    shutil.copy2(a.outdir/"baseline_denoised.png", a.outdir/"demo_baseline.png")
    print("Baseline demonstration complete:",a.outdir.resolve())
if __name__=="__main__": main()

