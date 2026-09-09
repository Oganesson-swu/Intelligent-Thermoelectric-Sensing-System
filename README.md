# Machine learning enabled thermoelectric leakage monitoring

This repository contains the Python code used for event-level dual-threshold leakage detection, feature-based liquid classification and learning-curve analysis. Peak features were extracted separately with the Peak Analyzer in OriginPro 2024. 

## Data

`data/processed_features.csv` is an Excel workbook exported with a `.csv` filename. The scripts detect this format automatically. It contains the columns `max_voltage`, `min_voltage`, `Area`, `AreaIntgP(%)`, `FWHM`, `Width`, `Height` and `Type`. The no-leakage class is labelled `No leakage`; its peak features are represented by `/` and are excluded automatically from liquid classification.

Running `code/split_data.py` creates `data/train.csv`, `data/validation.csv` and `data/independent_test.csv` using stratification and random seed 2026.

## Reproduction

From the repository root:

```bash
pip install -r requirements.txt
python code/split_data.py
python code/compare_models.py
python code/train_gradient_boosting.py
python code/learning_curve.py
python code/scan_dual_thresholds.py \
  --train-csv data/train.csv \
  --validation-csv data/validation.csv \
  --test-csv data/independent_test.csv \
  --max-column max_voltage --min-column min_voltage --label-column Type \
  --step 0.01 --a-min 0.015 --a-max 0.215 --b-min -0.125 --b-max 0.085 \
  --outdir analysis/threshold
python code/baseline_denoising.py
```

The dual-threshold scan treats a sample as no leakage only when it remains strictly between the thresholds, i.e. `threshold_B < min_voltage` and `max_voltage < threshold_A`. Candidate pairs are scanned every 0.01 V, and the `A <= B` region is retained as masked/invalid cells in the heatmap. The best pair is selected using the training plus validation data; the independent test split is used only for the final report.

## Final classifier

The final liquid classifier uses all five features without PCA and a `GradientBoostingClassifier` with 150 estimators, learning rate 0.05, maximum tree depth 2 and random seed 2026. StandardScaler is fitted within a pipeline.
The no-leakage class is used for threshold evaluation but is not a liquid class in the four-class classifier.

## Outputs

Analysis tables are written to `analysis/`. Figures are written to the external directory specified by the user when running `code/make_figures.py`; they are intentionally part of this repository.

## Software

Python 3.10 or newer, pandas, NumPy, SciPy, scikit-learn, Matplotlib and openpyxl. OriginPro 2024 is required only for reproducing the manual peak feature extraction step.
