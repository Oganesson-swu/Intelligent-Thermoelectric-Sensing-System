"""Grid search for dual thresholds used in thermoelectric leakage detection.

Expected CSV columns
--------------------
The script needs one maximum-voltage column and one minimum-voltage column.
Column names can be one of the common aliases below, for example:

    max_voltage,min_voltage,label
    0.125,-0.080,leakage
    0.035, 0.020,no_leakage

The label column is optional for scanning thresholds without ground truth,
but it is required for selecting the best threshold by accuracy/F1.

Detection rule used for event-level evaluation
-----------------------------------------------
For a response event:

    detected_leakage = (max_voltage >= threshold_A) OR
                       (min_voltage <= threshold_B)

with the physical constraint threshold_A > threshold_B.

Equivalently, a signal is classified as no leakage only when it remains
strictly between the two thresholds:

    threshold_B < min_voltage and max_voltage < threshold_A

The default grid is an explicitly bounded 0.01-voltage grid.  Explicit
bounds are important because a single outlier should not stretch the heatmap.

This corresponds to the two excursions illustrated in the manuscript:
crossing the upper threshold identifies an abnormal positive excursion and
crossing the lower threshold identifies the subsequent negative excursion.

Online state-machine rule
--------------------------
The optional state-machine function implements the manuscript's operational
logic for a time series:

    Normal: signal is between B and A
    Alarm:  signal rises above A
    Warn:   after Alarm, signal falls below B

For a CSV containing only event-level maxima/minima, event-level evaluation is
the appropriate use; the state machine requires the original time-resolved
signal curve.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from data_utils import read_table, is_no_leakage


MAX_ALIASES = [
    "max_voltage", "maximum_voltage", "max voltage", "maximum voltage",
    "max", "maximum", "v_max", "peak", "peak_height", "peak height",
    "最大值", "最大电压", "峰值", "峰高",
]
MIN_ALIASES = [
    "min_voltage", "minimum_voltage", "min voltage", "minimum voltage",
    "min", "minimum", "v_min", "最低值", "最小值", "最小电压",
]
LABEL_ALIASES = [
    "label", "class", "category", "target", "ground_truth", "ground truth",
    "状态", "类别", "标签", "是否泄漏", "泄漏状态",
]


def _normalise_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[\s\-_]+", "", text)
    return text


def find_column(df: pd.DataFrame, aliases: Iterable[str], explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Column {explicit!r} was not found. Available columns: {list(df.columns)}")
        return explicit

    normalised = {_normalise_name(c): c for c in df.columns}
    for alias in aliases:
        key = _normalise_name(alias)
        if key in normalised:
            return normalised[key]

    # Fallback: substring matching for names such as max_V or minimum signal.
    for c in df.columns:
        key = _normalise_name(c)
        if any(_normalise_name(alias) in key for alias in aliases):
            return c

    raise ValueError(
        "Could not identify a required column. "
        f"Available columns: {list(df.columns)}"
    )


def parse_label(value: object) -> Optional[bool]:
    """Convert common labels to True=leakage, False=no leakage."""
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value) in (0.0, 1.0):
            return bool(int(value))

    text = str(value).strip().lower()
    text = text.replace(" ", "").replace("_", "").replace("-", "")
    leakage = {"1", "true", "yes", "y", "leak", "leakage", "leaking", "abnormal", "alarm", "泄漏", "有泄漏", "异常"}
    no_leakage = {"0", "false", "no", "n", "noleak", "noleakage", "normal", "nonleakage", "无泄漏", "正常"}
    if text in leakage:
        return True
    if text in no_leakage:
        return False
    return None


def load_data(
    csv_path: Path,
    max_column: Optional[str] = None,
    min_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> tuple[pd.DataFrame, str, str, Optional[str]]:
    df = read_table(csv_path)
    max_col = find_column(df, MAX_ALIASES, max_column)
    min_col = find_column(df, MIN_ALIASES, min_column)

    df[max_col] = pd.to_numeric(df[max_col], errors="coerce")
    df[min_col] = pd.to_numeric(df[min_col], errors="coerce")
    df = df.dropna(subset=[max_col, min_col]).copy()

    label_col = None
    if label_column or any(_normalise_name(c) in {_normalise_name(a) for a in LABEL_ALIASES} for c in df.columns):
        label_col = find_column(df, LABEL_ALIASES, label_column)
        parsed = df[label_col].map(parse_label)
        if parsed.notna().all():
            df["_is_leakage"] = parsed.astype(bool)
        elif is_no_leakage(df[label_col]).any():
            # Multiclass source tables are valid for binary leak detection:
            # every named liquid is leakage and only the no-leakage label is
            # treated as the negative class.
            df["_is_leakage"] = ~is_no_leakage(df[label_col])
        else:
            bad = df.loc[parsed.isna(), label_col].drop_duplicates().tolist()
            raise ValueError(
                f"Unrecognised label values in {label_col!r}: {bad}. "
                "Use a no-leakage label or binary leakage/no_leakage labels."
            )

    return df, max_col, min_col, label_col


def make_threshold_grid(min_value: float, max_value: float, step: float = 0.01) -> np.ndarray:
    """Return a numerically stable, inclusive grid with the requested step."""
    if step <= 0 or max_value < min_value:
        raise ValueError("Require step > 0 and max_value >= min_value")
    n = int(np.floor((max_value - min_value) / step + 1e-9)) + 1
    grid = min_value + step * np.arange(n, dtype=float)
    # Avoid values such as 0.09999999999999999 in CSV/image labels.
    return np.round(grid, 10)


def evaluate_thresholds(
    df: pd.DataFrame,
    max_col: str,
    min_col: str,
    threshold_a: float,
    threshold_b: float,
) -> dict:
    """Evaluate one A/B pair. A is upper, B is lower, and A must be > B."""
    max_v = df[max_col].to_numpy(dtype=float)
    min_v = df[min_col].to_numpy(dtype=float)
    no_leakage = (min_v > threshold_b) & (max_v < threshold_a)
    predicted = ~no_leakage

    result = {
        "threshold_A": float(threshold_a),
        "threshold_B": float(threshold_b),
        "valid_physical_pair": bool(threshold_a > threshold_b),
        "n": int(len(df)),
        "predicted_leakage": int(predicted.sum()),
        "predicted_no_leakage": int((~predicted).sum()),
        "no_leakage_between_A_B": int(no_leakage.sum()),
    }

    if "_is_leakage" not in df:
        result.update({"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan,
                       "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan})
        return result

    actual = df["_is_leakage"].to_numpy(dtype=bool)
    tp = int(np.sum(predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    accuracy = (tp + tn) / len(actual) if len(actual) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    result.update({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                   "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return result


def grid_scan(
    df: pd.DataFrame,
    max_col: str,
    min_col: str,
    a_grid: np.ndarray,
    b_grid: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for threshold_a in a_grid:
        for threshold_b in b_grid:
            if threshold_a <= threshold_b:
                # Keep invalid pairs in the output so the heatmap shows the
                # triangular A<=B region explicitly as masked cells.
                rows.append({
                    "threshold_A": float(threshold_a),
                    "threshold_B": float(threshold_b),
                    "valid_physical_pair": False,
                    "n": int(len(df)),
                    "predicted_leakage": np.nan,
                    "predicted_no_leakage": np.nan,
                    "no_leakage_between_A_B": np.nan,
                    "accuracy": np.nan, "precision": np.nan,
                    "recall": np.nan, "f1": np.nan,
                    "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
                })
                continue
            rows.append(evaluate_thresholds(df, max_col, min_col, threshold_a, threshold_b))
    return pd.DataFrame(rows)


def plot_heatmap(results: pd.DataFrame, out_path: Path, criterion: str) -> None:
    """Create the manuscript-style Threshold A versus Threshold B heatmap."""
    import matplotlib.pyplot as plt

    pivot = results.pivot(index="threshold_A", columns="threshold_B", values=criterion)
    pivot = pivot.sort_index().sort_index(axis=1)
    values = pivot.to_numpy(dtype=float)
    a_values = pivot.index.to_numpy(dtype=float)
    b_values = pivot.columns.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=220)
    cmap = plt.get_cmap("RdPu").copy()
    cmap.set_bad("#222222")
    im = ax.imshow(
        masked, origin="lower", aspect="auto", cmap=cmap, vmin=0.80, vmax=1.0,
        extent=[b_values.min() - 0.005, b_values.max() + 0.005,
                a_values.min() - 0.005, a_values.max() + 0.005],
        interpolation="nearest",
    )
    # Highlight the requested high-performance region without cropping any
    # cells with accuracy >= 0.80 from the displayed coordinate range.
    finite = np.isfinite(values)
    if finite.any() and np.nanmin(values) <= 0.80 <= np.nanmax(values):
        ax.contour(b_values, a_values, np.where(finite, values, np.nan),
                   levels=[0.80], colors="white", linewidths=1.2)
    ax.set_xlabel("Threshold B (V)")
    ax.set_ylabel("Threshold A (V)")
    ax.set_title(f"Dual-threshold scan ({criterion})")
    ax.set_xlim(b_values.min() - 0.005, b_values.max() + 0.005)
    ax.set_ylim(a_values.min() - 0.005, a_values.max() + 0.005)
    fig.colorbar(im, ax=ax, label=criterion.replace("_", " ").title() + " (0.80–1.00)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def online_state_machine(signal: Iterable[float], threshold_a: float, threshold_b: float) -> list[str]:
    """Apply the manuscript's Normal -> Alarm -> Warn logic to a time series."""
    state = "Normal"
    states = []
    for value in signal:
        value = float(value)
        if state == "Normal" and value >= threshold_a:
            state = "Alarm"
        elif state == "Alarm" and value <= threshold_b:
            state = "Warn"
        elif state == "Normal" and value <= threshold_b:
            # A lower excursion without a preceding upper excursion is kept
            # as Normal here; change this branch if your firmware treats it
            # as an independent alarm.
            state = "Normal"
        states.append(state)
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid scan dual thresholds A/B for leakage detection")
    parser.add_argument("csv", type=Path, nargs="?", help="Input CSV (used when split files are not supplied)")
    parser.add_argument("--train-csv", type=Path, help="Training split used for threshold selection")
    parser.add_argument("--validation-csv", type=Path, help="Validation split used for threshold selection")
    parser.add_argument("--test-csv", type=Path, help="Independent test split for final evaluation")
    parser.add_argument("--outdir", type=Path, default=Path("dual_threshold_results"))
    parser.add_argument("--max-column", default=None, help="Maximum-voltage column name")
    parser.add_argument("--min-column", default=None, help="Minimum-voltage column name")
    parser.add_argument("--label-column", default=None, help="Ground-truth column; leakage vs no_leakage")
    parser.add_argument("--step", type=float, default=0.01, help="Threshold grid step (default: 0.01 V)")
    # The default ranges are centred approximately on the optimum found for
    # the supplied dataset (A=0.09 V, B=0.02 V), while retaining negative
    # voltage values on both axes. Users can override these bounds explicitly.
    parser.add_argument("--a-min", type=float, default=0.015)
    parser.add_argument("--a-max", type=float, default=0.215)
    parser.add_argument("--b-min", type=float, default=-0.125)
    parser.add_argument("--b-max", type=float, default=0.085)
    parser.add_argument("--criterion", choices=["accuracy", "f1", "balanced_accuracy"], default="accuracy")
    args = parser.parse_args()

    if args.train_csv or args.validation_csv or args.test_csv:
        if not (args.train_csv and args.validation_csv and args.test_csv):
            parser.error("--train-csv, --validation-csv and --test-csv must be supplied together")
        train, max_col, min_col, label_col = load_data(args.train_csv, args.max_column, args.min_column, args.label_column)
        valid, _, _, _ = load_data(args.validation_csv, max_col, min_col, label_col)
        test, _, _, _ = load_data(args.test_csv, max_col, min_col, label_col)
        selection_df = pd.concat([train, valid], ignore_index=True)
        df = pd.concat([selection_df.assign(split="train_validation"), test.assign(split="independent_test")], ignore_index=True)
    else:
        if args.csv is None:
            parser.error("provide csv or all three split files")
        selection_df, max_col, min_col, label_col = load_data(args.csv, args.max_column, args.min_column, args.label_column)
        test = None
        df = selection_df
    args.outdir.mkdir(parents=True, exist_ok=True)

    a_grid = make_threshold_grid(args.a_min, args.a_max, args.step)
    b_grid = make_threshold_grid(args.b_min, args.b_max, args.step)
    results = grid_scan(selection_df, max_col, min_col, a_grid, b_grid)

    if "_is_leakage" in df:
        # Balanced accuracy is calculated after the scan to avoid privileging
        # the majority class when leakage and no-leakage counts differ.
        results["balanced_accuracy"] = (
            0.5 * (results["tp"] / (results["tp"] + results["fn"]).replace(0, np.nan)
                   + results["tn"] / (results["tn"] + results["fp"]).replace(0, np.nan))
        ).fillna(0.0)
        results = results.sort_values(
            by=[args.criterion, "f1", "accuracy", "threshold_A"],
            ascending=[False, False, False, True],
        )
        best = results.iloc[0].to_dict()
        if test is not None:
            test_result = evaluate_thresholds(test, max_col, min_col, best["threshold_A"], best["threshold_B"])
            test_result["split"] = "independent_test"
            with open(args.outdir / "independent_test_result.json", "w", encoding="utf-8") as f:
                json.dump(test_result, f, ensure_ascii=False, indent=2, default=lambda x: None if pd.isna(x) else x)
    else:
        # Without labels, report candidate pairs but do not pretend to know
        # which pair is optimal. Prefer a conservative pair near the median.
        best = {
            "threshold_A": float(np.median(a_grid)),
            "threshold_B": float(np.median(b_grid)),
            "note": "No ground-truth label column supplied; no accuracy-based optimum exists.",
        }

    results.to_csv(args.outdir / "threshold_grid_results.csv", index=False)
    with open(args.outdir / "best_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2, default=lambda x: None if pd.isna(x) else x)

    # Save a compact table with the source-column mapping and data summary.
    summary = {
        "input_csv": str(args.csv),
        "n_rows_used_for_selection": int(len(selection_df)),
        "n_rows_independent_test": int(len(test)) if test is not None else None,
        "max_column": max_col,
        "min_column": min_col,
        "label_column": label_col,
        "step": args.step,
        "a_range": [args.a_min, args.a_max],
        "b_range": [args.b_min, args.b_max],
        "criterion": args.criterion,
        "best_thresholds": best,
    }
    with open(args.outdir / "scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda x: None if pd.isna(x) else x)

    print("Input rows used:", len(df))
    print("Detected columns:", {"max": max_col, "min": min_col, "label": label_col})
    print("Results:", args.outdir / "threshold_grid_results.csv")
    print("Best thresholds:", args.outdir / "best_thresholds.json")
    plot_heatmap(results, args.outdir / "dual_threshold_accuracy_heatmap.png", args.criterion)
    print("Heatmap:", args.outdir / "dual_threshold_accuracy_heatmap.png")
    print(json.dumps(best, ensure_ascii=False, indent=2, default=lambda x: None if pd.isna(x) else x))


if __name__ == "__main__":
    main()
