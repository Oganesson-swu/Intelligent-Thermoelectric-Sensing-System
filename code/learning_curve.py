"""Learning curve for five-feature Gradient Boosting, excluding No leakage."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_utils import load_features, liquid_only, FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "learning_curve"
SEED, REPEATS, STEP = 2026, 10, 50

def make_model(seed):
    return Pipeline([("scaler", StandardScaler()), ("classifier", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=seed))])

def subset(df, n, seed):
    rng=np.random.default_rng(seed); parts=[]
    for label, g in df.groupby(TARGET): parts.append(g.loc[rng.choice(g.index.to_numpy(), n, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def score(m, df):
    y=df[TARGET]; p=m.predict(df[FEATURES])
    return {"accuracy":accuracy_score(y,p),"balanced_accuracy":balanced_accuracy_score(y,p),"macro_f1":f1_score(y,p,average="macro",zero_division=0)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=liquid_only(load_features(DATA/"train.csv")); validation=liquid_only(load_features(DATA/"validation.csv")); test=liquid_only(load_features(DATA/"independent_test.csv"))
    max_n=train.groupby(TARGET).size().min(); rows=[]
    for n in range(STEP,max_n+1,STEP):
        for repeat in range(REPEATS):
            sampling_seed=SEED+n*100+repeat; m=make_model(SEED+repeat); s=subset(train,n,sampling_seed); m.fit(s[FEATURES],s[TARGET])
            for split,df in [("validation",validation),("independent_test",test)]: rows.append({"samples_per_class":n,"training_samples":n*df[TARGET].nunique(),"repeat":repeat+1,"split":split,**score(m,df)})
    raw=pd.DataFrame(rows); raw.to_csv(OUT/"raw_results.csv",index=False,encoding="utf-8-sig")
    summary=raw.groupby(["samples_per_class","split"],as_index=False).agg(repeats=("repeat","count"),accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),balanced_accuracy_mean=("balanced_accuracy","mean"),balanced_accuracy_std=("balanced_accuracy","std"),macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"))
    summary.to_csv(OUT/"summary.csv",index=False,encoding="utf-8-sig")
    fig,axes=plt.subplots(1,3,figsize=(15,5),dpi=200); metrics=["accuracy","balanced_accuracy","macro_f1"]
    colors={"validation":"#377eb8","independent_test":"#e41a1c"}
    for ax,metric in zip(axes,metrics):
        for split in colors:
            q=summary[summary.split==split].sort_values("samples_per_class"); ax.plot(q.samples_per_class,q[f"{metric}_mean"],marker="o",label=split,color=colors[split]); ax.fill_between(q.samples_per_class.to_numpy(),(q[f"{metric}_mean"]-q[f"{metric}_std"]).to_numpy(),(q[f"{metric}_mean"]+q[f"{metric}_std"]).to_numpy(),alpha=.12,color=colors[split])
        ax.set_xlabel("Training samples per liquid"); ax.set_ylabel(metric.replace("_"," ").title()); ax.set_ylim(.5,1.01); ax.grid(alpha=.25)
    axes[0].legend(frameon=False); fig.suptitle("Learning curve: five-feature Gradient Boosting"); fig.tight_layout(); fig.savefig(OUT/"learning_curve.png",bbox_inches="tight"); plt.close(fig)
    print(summary.to_string(index=False)); print("Output:",OUT)
if __name__ == "__main__": main()
