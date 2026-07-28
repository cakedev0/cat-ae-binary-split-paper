# Binary Split

This repository accompanies the paper *"Binary Split Categorical feature with Mean
Absolute Error Criteria in CART"* (`paper.pdf`). It contains a C++ implementation of the
paper's exact MAE binary-split algorithm, along with a Python script that reproduces the
paper's experiments (Table 2) against scikit-learn and LightGBM baselines.

**Note on this version:** the paper's own experiments predate scikit-learn's native
categorical-feature support and compare against a manual median-encoding trick (the
"heuristic" discussed in the paper, `O(n^2)` and not guaranteed optimal). This repo has
been updated to instead use `DecisionTreeRegressor(categorical_features=...)`, added in
scikit-learn nightly (>=1.10.dev), which searches for the optimal categorical grouping
directly rather than through an encoding. It caps out at 255 categories, so features
above that still fall back to the median-sorted ordinal encoding. See `gen.py`'s
`sklearn_solution` for details.

## File Overview

- `binary_split.cpp`: C++ implementation of the paper's algorithm.
- `gen.py`: Python script to:
  - Load and preprocess each dataset from the paper (see `DATASETS` registry),
  - Run the algorithm via the compiled C++ executable,
  - Execute scikit-learn and LightGBM baselines,
  - Collect and format results, per-dataset and combined.

## Input Format for C++ Executable

The program expects input in the following format:
```
n k
m₁ x₁₁ x₁₂ ... x₁ₘ₁
m₂ x₂₁ x₂₂ ... x₂ₘ₂
...
mₖ xₖ₁ xₖ₂ ... xₖₘₖ
```

Where:
- `n` = total number of points,
- `k` = number of categories (sets),
- Each of the following `k` lines contains the number of elements in a category and their coordinates.

## Building the C++ Executable

The checked-in `binary_split.exe` is a Linux ELF binary (recompiled from Windows PE).
Rebuild it with:

```bash
g++ -O2 -std=c++17 -o binary_split.exe binary_split.cpp
```

## Datasets

Datasets are **not** committed to this repo (only `dataset_42225.arff` ships as a small
example). Download the ones you want to run and place them in the repo root under the
filenames below, then set `DATASETS_TO_RUN` in `gen.py` accordingly.

| `DATASETS` key            | File to save as   | Source | Download |
|---------------------------|--------------------|--------|----------|
| `diamonds`                 | `dataset_42225.arff` (included) | OpenML 42225 | already in repo |
| `gpu_kernel_performance`   | `dataset_45662.arff` | OpenML 45662 | `curl -L "https://openml.org/data/v1/download/22117147/simulated_sgemm_gpu_kernel_performance.arff" -o dataset_45662.arff` |
| `house_sales`              | `dataset_42731.arff` | OpenML 42731 | `curl -L "https://openml.org/data/v1/download/22044765/house_sales.arff" -o dataset_42731.arff` |
| `boston`                   | `dataset_531.arff`   | OpenML 531   | `curl -L "https://openml.org/data/v1/download/52643/boston.arff" -o dataset_531.arff` |
| `delays_zurich_transport`  | `dataset_40753.arff` | OpenML 40753 | `curl -L "https://openml.org/data/v1/download/5698591/delays_zurich_transport.arff" -o dataset_40753.arff` |
| `wine`                     | `WineQT.csv`          | Kaggle [yasserh/wine-quality-dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) | `kaggle datasets download -d yasserh/wine-quality-dataset -p . --unzip` (requires a [Kaggle API token](https://www.kaggle.com/docs/api)) |
| `predict_droughts`         | `predict_droughts.csv` (not verified) | Kaggle [cdminix/us-drought-meteorological-data](https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data) | not wired up — target column is a guess; download and adjust `DATASETS['predict_droughts']` in `gen.py` before use |

Notes:
- `delays_zurich_transport` (OpenML 40753) has string-typed columns that
  `scipy.io.arff.loadarff` can't parse; `gen.py` handles this automatically via a small
  hand-rolled ARFF header parser (`_load_arff_mixed`), no manual CSV conversion needed.
- `predict_droughts` is excluded from `DATASETS_TO_RUN` by default: it's large
  (19.3M rows), Kaggle-hosted, and its target column hasn't been verified against the
  actual file.

## How to Run

1. Download the datasets you want (see table above) into the repo root.
2. Build `binary_split.exe` (see above).
3. Edit `DATASETS_TO_RUN` near the bottom of `gen.py` to pick which datasets to run.
4. Run:

```bash
python gen.py
```

## Output

- `result.txt`: Detailed runtime logs and MAE scores for each feature and method, across all datasets run.
- `{test_name}.md` / `.xlsx` per dataset (e.g. `diamonds_42225.md`): a summary table comparing scikit-learn, LightGBM, and the paper's C++ algorithm for each feature of that dataset.
- `table2_reproduction.md` / `.xlsx`: combined table across every dataset run in the same call, in the same layout as the paper's Table 2 (with an added `dataset` column).

## Customization Options

- Modify `now_n` inside `run_dataset()` in `gen.py` to try different data sizes instead of the full dataset.
- Uncomment the `draw_figure()` / `draw_table()` calls (commented out inside `run_dataset()`) to also get per-scale runtime plots/tables.

## ARFF Format Compatibility Notes

- **OpenML ID 42225** (diamonds): the raw ARFF needs `'Very Good'` replaced with `Very Good` (quotes removed) to avoid parser errors — already fixed in the committed `dataset_42225.arff`.
- **OpenML ID 40753** (delays_zurich_transport): handled automatically by `_load_arff_mixed` in `gen.py`, no manual conversion needed (see Notes above).

## Python Dependencies

```bash
pip install numpy pandas matplotlib lightgbm scipy tabulate openpyxl
```

Plus a nightly build of scikit-learn (>=1.10.dev) for `DecisionTreeRegressor(categorical_features=...)`:

```bash
pip install --pre --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple scikit-learn
```

## License

This project is intended for academic research only. Please contact us for other uses.
