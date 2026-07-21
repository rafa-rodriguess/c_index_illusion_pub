# Domain 3 — external code / documents

## Author code (public)

- GitHub: https://github.com/habedi/SurvivalAnalysisQACommunities
- Key notebook: `survival analysis using RSF.ipynb`
- Data: `processed_data/{p,ds,cs}/user_features.csv` (tab-separated) — installed by `B02_download_stackexchange.py`
- Author fold C-indices: `data/raw/stackexchange/author_results/` (LFS `.bin`); see `AUTHOR_BINS.md`

## Protocol extracted from author notebook + paper

| Item | Value |
|------|--------|
| Duration | `MonthsActive` |
| Event | `MonthsSinceLastActivity > θ` → disengaged=1 |
| θ | 24 and 36 months |
| Feature sets | behavioural (A1–A5), content (A6–A11), combined |
| Cohort | Paper §5.2: `Q ∪ A ∪ C ∪ U ∪ D` (any of Q/A/C/Up/Down counts > 0) |
| CV | Drop 1% holdout, then 5-fold **without shuffle** × 30 runs |
| RSF | `RandomSurvivalForestModel(num_trees=5)`; `max_features=sqrt`, `max_depth=5`, `min_node_size=30`, `sample_size_pct=0.63` |
| Backend | PySurvival + `pysurvival.utils.metrics.concordance_index` |

## Our runtime (PySurvival) — outsider setup

PySurvival 0.1.2 does not build cleanly on modern macOS/Python (Cython `tp_print` ABI). Use a **dedicated** conda env (Python 3.9) with a one-line patch on the generated C++ (`tp_print` assignments commented out).

### Create the env (once)

```bash
conda create -n d3-pysurvival python=3.9 -y
conda activate d3-pysurvival
pip install "numpy<1.24" "pandas<2" scikit-learn==0.22.2 scipy pyarrow
# Install pysurvival 0.1.2 from source; if the build fails on tp_print,
# edit the generated .cpp under the build tree (comment out every
# `->tp_print =` line) and re-run pip install.
pip install pysurvival==0.1.2
```

Point the main trunk at this interpreter:

```bash
# preferred: ./bootstrap_envs.sh --d3
export CDS_D3_PYTHON="$(conda info --base)/envs/d3-pysurvival/bin/python"
# or absolute path, e.g. $HOME/miniconda3/envs/d3-pysurvival/bin/python
```

`00_pipeline.ipynb` / `src.repro.resolve_script_python()` use **`.venv`** for almost every stage and **`CDS_D3_PYTHON`** only for `D02_DOMAIN_03_train_rsf.py`.

### Run DOMAIN_03 train

```bash
# Features / CV / gap: main .venv is fine
python -W default D00_DOMAIN_03_features_stackexchange.py
python -W default D01_DOMAIN_03_cv_protocol.py
# Table-8 RSF fits: must use d3-pysurvival
"$CDS_D3_PYTHON" -W default D02_DOMAIN_03_train_rsf.py
# Ladder joblibs for Block E: main .venv (sksurv), not PySurvival
python -W default D02b_DOMAIN_03_export_ladder_rsf.py
python -W default D03_DOMAIN_03_gap.py
```

## Author `.bin` takeaway

Pickles store `(variable_importance, c_index)` per `(run, fold)` only — **no hyperparameters**. Bin means match Table 8 to ~2 decimals (see `AUTHOR_BINS.md`).
