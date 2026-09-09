"""Evaluate Gradient Boosting using only the Height feature on data split 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_utils import load_features, liquid_only, TARGET


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "analysis" / "height_only_baseline"
FEATURE = "Height"
TARGET = "Type"


def load_data(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    df = load_features(path)
    df = liquid_only(df)
    required = {FEATURE, TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    df = df[[FEATURE, TARGET]].copy()
    df[FEATURE] = pd.to_numeric(df[FEATURE], errors="coerce")
    if df[FEATURE].isna().any():
        raise ValueError(f"{path.name} contains non-numeric or missing Height values")
    return df.reset_index(drop=True)


def make_model() -> Pipeline:
    # Match the optimized model-comparison protocol: standardization, no PCA,
    # and the same Gradient Boosting hyperparameters used in model comparison.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=2026,
                ),
            ),
        ]
    )


def metrics_row(y_true: pd.Series, y_pred: np.ndarray, split: str) -> dict:
    return {
        "feature": FEATURE,
        "pca": "No_PCA",
        "model": "Gradient_boosting",
        "split": split,
        "n_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, labels, filename: str, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(
        cm,
        index=[f"Actual_{label}" for label in labels],
        columns=[f"Predicted_{label}" for label in labels],
    ).to_csv(OUT_DIR / filename.replace(".png", ".csv"), encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12,
            )
    fig.colorbar(image, ax=ax, label="Number of samples")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_data("train.csv")
    validation = load_data("validation.csv")
    test = load_data("independent_test.csv")
    labels = sorted(set(train[TARGET]) | set(validation[TARGET]) | set(test[TARGET]))

    model = make_model()
    model.fit(train[[FEATURE]], train[TARGET])
    validation_pred = model.predict(validation[[FEATURE]])

    train_validation = pd.concat([train, validation], ignore_index=True)
    final_model = make_model()
    final_model.fit(train_validation[[FEATURE]], train_validation[TARGET])
    test_pred = final_model.predict(test[[FEATURE]])

    rows = [
        metrics_row(validation[TARGET], validation_pred, "cross_validation"),
        metrics_row(test[TARGET], test_pred, "independent_test_final"),
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "height_gradient_boosting_metrics.csv", index=False, encoding="utf-8-sig")

    for df, pred, split in [
        (validation, validation_pred, "cross_validation"),
        (test, test_pred, "independent_test_final"),
    ]:
        predictions = df.copy()
        predictions.insert(0, "split", split)
        predictions["predicted_Type"] = pred
        predictions["correct"] = predictions[TARGET].to_numpy() == pred
        predictions.to_csv(OUT_DIR / f"height_gradient_boosting_{split}_predictions.csv", index=False, encoding="utf-8-sig")

    report = classification_report(
        test[TARGET], test_pred, labels=labels, target_names=labels,
        zero_division=0, output_dict=True
    )
    pd.DataFrame(report).T.to_csv(
        OUT_DIR / "height_gradient_boosting_independent_test_classification_report.csv",
        encoding="utf-8-sig",
    )
    save_confusion_matrix(
        test[TARGET],
        test_pred,
        labels,
        "height_gradient_boosting_independent_test_confusion_matrix.png",
        "Height-only Gradient Boosting: Independent Test",
    )

    print("Feature:", FEATURE)
    print("Model: Gradient Boosting (No PCA)")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()
