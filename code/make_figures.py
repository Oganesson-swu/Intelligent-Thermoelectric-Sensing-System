"""Generate reproducibility figures into an external output directory."""
from pathlib import Path
import argparse
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--outdir",type=Path,required=True); a=p.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    comp=ROOT/"analysis"/"model_comparison"/"model_comparison.csv"
    if comp.exists():
        d=pd.read_csv(comp); q=d[d["split"]=="independent_test"].sort_values("balanced_accuracy"); fig,ax=plt.subplots(figsize=(8,4),dpi=200); ax.barh(q["model"],q["balanced_accuracy"],color="#4c78a8"); ax.set_xlim(0,1); ax.set_xlabel("Independent-test balanced accuracy"); ax.set_title("Nine-classifier comparison using all five features"); fig.tight_layout(); fig.savefig(a.outdir/"model_comparison_accuracy.png",bbox_inches="tight"); plt.close(fig)
    lc=ROOT/"analysis"/"learning_curve"/"learning_curve.png"
    if lc.exists(): shutil.copy2(lc,a.outdir/"learning_curve.png")
    demo=ROOT/"analysis"/"baseline_example"/"demo_baseline.png"
    if demo.exists(): shutil.copy2(demo,a.outdir/"baseline_demo.png")
    th=ROOT/"analysis"/"threshold"/"threshold_grid_results.csv"
    if th.exists():
        d=pd.read_csv(th)
        piv=d.pivot(index="threshold_A",columns="threshold_B",values="accuracy").sort_index().sort_index(axis=1)
        vals=np.ma.masked_invalid(piv.to_numpy(dtype=float))
        av=piv.index.to_numpy(dtype=float); bv=piv.columns.to_numpy(dtype=float)
        fig,ax=plt.subplots(figsize=(7.2,5.8),dpi=220)
        cmap=plt.get_cmap("RdPu").copy(); cmap.set_bad("#222222")
        im=ax.imshow(vals,origin="lower",aspect="auto",cmap=cmap,vmin=0.80,vmax=1.0,
                     extent=[bv.min()-0.005,bv.max()+0.005,av.min()-0.005,av.max()+0.005],
                     interpolation="nearest")
        ax.set_xlabel("Threshold B (V)"); ax.set_ylabel("Threshold A (V)")
        raw=piv.to_numpy(dtype=float)
        finite=np.isfinite(raw)
        if finite.any() and np.nanmin(raw) <= .8 <= np.nanmax(raw):
            ax.contour(bv,av,np.where(finite,raw,np.nan),levels=[.8],colors="white",linewidths=1.2)
        ax.set_title("Dual-threshold scan (accuracy ≥ 0.80 highlighted)")
        fig.colorbar(im,ax=ax,label="Accuracy (0.80–1.00)")
        fig.tight_layout(); fig.savefig(a.outdir/"dual_threshold_accuracy_heatmap.png",bbox_inches="tight"); plt.close(fig)
    for f in [ROOT/"analysis"/"classification"/"confusion_matrix.png"]:
        if f.exists(): shutil.copy2(f,a.outdir/"final_model_confusion_matrix.png")
    print("Figures:",a.outdir.resolve())
if __name__=="__main__": main()
