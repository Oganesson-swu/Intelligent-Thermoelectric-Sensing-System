"""Shared data loading utilities for the reproducibility repository."""
from pathlib import Path
import pandas as pd

FEATURES = ["Area", "AreaIntgP(%)", "FWHM", "Width", "Height"]
THRESHOLD_COLUMNS = ["max_voltage", "min_voltage"]
TARGET = "Type"
NO_LEAKAGE_LABELS = {"no leakage", "no_leakage", "noleakage", "normal", "0", "false", "无泄漏", "正常"}


def read_table(path: Path, sheet_name=0) -> pd.DataFrame:
    """Read normal CSV or an Excel workbook saved with a .csv extension."""
    # Some Origin exports retain an Excel workbook internally while carrying
    # a .csv filename. Try Excel first, then fall back to a real CSV.
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")


def load_features(path: Path, include_threshold=False) -> pd.DataFrame:
    df = read_table(path)
    required = FEATURES + [TARGET] + (THRESHOLD_COLUMNS if include_threshold else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}; available: {list(df.columns)}")
    out = df[required].copy()
    for col in FEATURES + (THRESHOLD_COLUMNS if include_threshold else []):
        out[col] = pd.to_numeric(out[col].replace({"/": pd.NA, "": pd.NA}), errors="coerce")
    return out


def is_no_leakage(series: pd.Series) -> pd.Series:
    normalised = (series.astype(str).str.strip().str.lower()
                  .str.replace(" ", "", regex=False)
                  .str.replace("_", "", regex=False)
                  .str.replace("-", "", regex=False))
    return normalised.isin({x.replace(" ", "").replace("_", "").replace("-", "") for x in NO_LEAKAGE_LABELS})


def liquid_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~is_no_leakage(df[TARGET])].reset_index(drop=True)
