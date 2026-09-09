"""Train the final five-feature Gradient Boosting liquid classifier."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_utils import load_features, liquid_only, FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "classification"
SEED = 2026

def make_model():
    return Pipeline([("scaler", StandardScaler()), ("classifier", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=SEED))])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = liquid_only(load_features(DATA / "train.csv")); validation = liquid_only(load_features(DATA / "validation.csv")); test = liquid_only(load_features(DATA / "independent_test.csv"))
    m = make_model(); m.fit(train[FEATURES], train[TARGET]); vp = m.predict(validation[FEATURES])
    train_validation = pd.concat([train, validation], ignore_index=True)
    final = make_model(); final.fit(train_validation[FEATURES], train_validation[TARGET]); tp = final.predict(test[FEATURES])
    rows=[]
    for name, y, p in [("validation", validation[TARGET], vp), ("independent_test", test[TARGET], tp)]:
        rows.append({"split":name,"n_samples":len(y),"accuracy":accuracy_score(y,p),"balanced_accuracy":balanced_accuracy_score(y,p),"macro_f1":f1_score(y,p,average="macro",zero_division=0)})
    pd.DataFrame(rows).to_csv(OUT / "final_model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(classification_report(test[TARGET],tp,output_dict=True,zero_division=0)).T.to_csv(OUT / "classification_report.csv", encoding="utf-8-sig")
    labels=sorted(test[TARGET].unique()); cm=confusion_matrix(test[TARGET],tp,labels=labels)
    pd.DataFrame(cm,index=[f"Actual_{x}" for x in labels],columns=[f"Predicted_{x}" for x in labels]).to_csv(OUT / "confusion_matrix.csv",encoding="utf-8-sig")
    fig,ax=plt.subplots(figsize=(7,6),dpi=220); im=ax.imshow(cm,cmap="Blues"); ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_xticks(range(len(labels)),labels,rotation=30,ha="right"); ax.set_yticks(range(len(labels)),labels)
    for i in range(len(labels)):
        for j in range(len(labels)): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.colorbar(im,ax=ax,label="Number of samples"); fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png",bbox_inches="tight"); plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False)); print("Output:",OUT)
if __name__ == "__main__": main()
