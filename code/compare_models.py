"""Compare candidate classifiers without PCA on all five features."""
from pathlib import Path
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from data_utils import load_features, liquid_only, FEATURES, TARGET

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"; OUT=ROOT/"analysis"/"model_comparison"; SEED=2026

def models():
    return {"Logistic_regression":LogisticRegression(max_iter=5000,random_state=SEED),"SVC_RBF":SVC(C=1.0,gamma="scale",kernel="rbf"),"Random_forest":RandomForestClassifier(n_estimators=300,min_samples_leaf=2,random_state=SEED,n_jobs=-1),"Gradient_boosting":GradientBoostingClassifier(n_estimators=150,learning_rate=.05,max_depth=2,random_state=SEED)}
def pipe(m): return Pipeline([("scaler",StandardScaler()),("model",m)])
def main():
    OUT.mkdir(parents=True,exist_ok=True); tr=liquid_only(load_features(DATA/"train.csv")); va=liquid_only(load_features(DATA/"validation.csv")); te=liquid_only(load_features(DATA/"independent_test.csv")); rows=[]
    for name,temp in models().items():
        m=pipe(clone(temp)); m.fit(tr[FEATURES],tr[TARGET]); p=m.predict(va[FEATURES]); rows.append({"model":name,"split":"validation","accuracy":accuracy_score(va[TARGET],p),"balanced_accuracy":balanced_accuracy_score(va[TARGET],p),"macro_f1":f1_score(va[TARGET],p,average="macro")})
        m=pipe(clone(temp)); fit=pd.concat([tr,va]); m.fit(fit[FEATURES],fit[TARGET]); p=m.predict(te[FEATURES]); rows.append({"model":name,"split":"independent_test","accuracy":accuracy_score(te[TARGET],p),"balanced_accuracy":balanced_accuracy_score(te[TARGET],p),"macro_f1":f1_score(te[TARGET],p,average="macro")})
    pd.DataFrame(rows).to_csv(OUT/"model_comparison.csv",index=False,encoding="utf-8-sig"); print(pd.DataFrame(rows).to_string(index=False)); print("Output:",OUT)
if __name__=="__main__": main()
