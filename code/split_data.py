"""Create stratified 60/20/20 train, validation and independent-test CSV files."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from data_utils import load_features, TARGET

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed_features.csv"
OUT = ROOT / "data"
SEED = 2026


def main():
    df = load_features(INPUT, include_threshold=True)
    df.insert(0, "sample_id", range(1, len(df) + 1))
    train, remainder = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=SEED)
    validation, test = train_test_split(remainder, test_size=0.5, stratify=remainder[TARGET], random_state=SEED)
    for name, part in [("train", train), ("validation", validation), ("independent_test", test)]:
        part.sort_values("sample_id").to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"Input: {len(df)}; train: {len(train)}; validation: {len(validation)}; test: {len(test)}")
    print(pd.concat([train.assign(split="train"), validation.assign(split="validation"), test.assign(split="independent_test")]).groupby(["split", TARGET]).size())


if __name__ == "__main__":
    main()
