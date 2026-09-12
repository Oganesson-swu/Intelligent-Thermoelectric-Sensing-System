"""Compare nine classifiers using all five extracted features."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_utils import FEATURES, TARGET, liquid_only, load_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "model_comparison"
SEED = 2026


def models() -> dict[str, object]:
    return {
        "Random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "Gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=SEED),
        "KNN_k5": KNeighborsClassifier(n_neighbors=5, weights="uniform"),
        "KNN_k11_distance": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "Decision_tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "SVC_RBF": SVC(C=1.0, gamma="scale", kernel="rbf"),
        "SVC_linear": SVC(C=1.0, kernel="linear"),
        "Logistic_regression": LogisticRegression(C=1.0, max_iter=5000, random_state=SEED),
        "LDA": LinearDiscriminantAnalysis(),
    }


def pipeline(model: object) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def metric_row(y_true: pd.Series, prediction: object) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, prediction, average="macro", zero_division=0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = liquid_only(load_features(DATA / "train.csv"))
    validation = liquid_only(load_features(DATA / "validation.csv"))
    test = liquid_only(load_features(DATA / "independent_test.csv"))
    rows: list[dict[str, object]] = []
    for name, template in models().items():
        model = pipeline(clone(template))
        model.fit(train[FEATURES], train[TARGET])
        row = {"model": name, "split": "validation", "n_samples": len(validation), "features": " + ".join(FEATURES)}
        row.update(metric_row(validation[TARGET], model.predict(validation[FEATURES])))
        rows.append(row)

        train_validation = pd.concat([train, validation], ignore_index=True)
        model = pipeline(clone(template))
        model.fit(train_validation[FEATURES], train_validation[TARGET])
        row = {"model": name, "split": "independent_test", "n_samples": len(test), "features": " + ".join(FEATURES)}
        row.update(metric_row(test[TARGET], model.predict(test[FEATURES])))
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    summary = results.groupby("model", as_index=False)[["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall"]].mean().sort_values("balanced_accuracy", ascending=False)
    summary.to_csv(OUT / "model_comparison_summary.csv", index=False, encoding="utf-8-sig")
    for split in ["validation", "independent_test"]:
        subset = results[results["split"] == split].sort_values("balanced_accuracy")
        fig, ax = plt.subplots(figsize=(8, 5), dpi=220)
        ax.barh(subset["model"], subset["balanced_accuracy"], color="#4C78A8")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Balanced accuracy")
        ax.set_title(f"Nine-classifier comparison: {split.replace('_', ' ').title()}")
        for y, value in enumerate(subset["balanced_accuracy"]):
            ax.text(value + 0.005, y, f"{value:.3f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"model_comparison_{split}.png", bbox_inches="tight")
        plt.close(fig)
    print("Features:", FEATURES)
    print("Classifiers:", list(models()))
    print(results.to_string(index=False))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
