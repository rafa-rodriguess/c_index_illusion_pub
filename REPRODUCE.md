# Reproduce — C-index Illusion (public)

Clone-and-run guide for the scientific pipeline (Cox / XGB / RSF + anchor ladder + paper assets).

Public repo: https://github.com/rafa-rodriguess/c_index_illusion_pub

## Two environments (intentional)

| Env | Role | Created by |
|-----|------|------------|
| **`.venv`** | Main trunk — Blocks A–H, P, D02b, AN, … | `A02` / `./bootstrap_envs.sh` |
| **`d3-pysurvival`** (conda) | Only `D02_DOMAIN_03_train_rsf.py` (PySurvival 0.1.2) | **`A02b`** (notebook) / `./bootstrap_envs.sh --d3` |

They are **not** mergeable: PySurvival needs Python 3.9 + a build patch; the main lock targets 3.11–3.13 + torch/sksurv.

The notebook / `src.repro.resolve_script_python()` pick the right interpreter per script.

## Prerequisites

| Requirement | Why |
|-------------|-----|
| Python 3.11–3.13 (author used 3.13.5) | Main trunk |
| Disk: tens of GB free | Backblaze Block B (~31 zip archives) |
| [Kaggle account](https://www.kaggle.com/) + API token | **Required** for Bondora (`B01`) |
| Graphviz (`dot` on `PATH`) | F01 workflow PDF in Block P |
| Optional: conda | Domain-3 Table-8 RSF (`--d3`) |

## 1. Bootstrap + activate

```bash
git clone https://github.com/rafa-rodriguess/c_index_illusion_pub.git
cd c_index_illusion_pub

./bootstrap_envs.sh          # creates .venv + installs requirements.lock
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Domain-3 is also created from the notebook (A02b). CLI equivalent:
./bootstrap_envs.sh --d3
# export CDS_D3_PYTHON=... is optional — resolvers probe conda + results/logs/d3_python_path.txt

python -W default A00_config.py
python -W default A01_setup_dirs.py
python -W default A03_env_check.py
```

From the notebook alone: run Block A top-to-bottom — **A02** creates `.venv`, **A02b** creates `d3-pysurvival` (needs `conda` on `PATH`).

### Kaggle credentials (required before B01)

```bash
mkdir -p ~/.kaggle
# place kaggle.json from https://www.kaggle.com/settings → API → Create New Token
chmod 600 ~/.kaggle/kaggle.json
# or: export KAGGLE_USERNAME=... KAGGLE_KEY=...
```

Dataset: `marcobeyer/bondora-p2p-loans` → `data/raw/bondora/LoanData.csv`.

## 2. Domain-3 RSF details

Full patch notes if `pip install pysurvival==0.1.2` fails: [`domain3-abedi-2022/CODE_ACCESS.md`](domain3-abedi-2022/CODE_ACCESS.md).

Ladder joblibs for Block E come from **`D02b_DOMAIN_03_export_ladder_rsf.py`** (sksurv; **main** `.venv`) — the notebook runs it after D02.

## 3. Run the pipeline

**Canonical:** open `00_pipeline.ipynb` with the **`.venv` kernel** and run top-to-bottom.

Individual steps:

```bash
python -W default C00_preregister_protocol.py
# … Blocks B–H …
python -W default P00_collect_artifacts.py
```

Expect multi-hour to multi-day wall clock on a laptop (Backblaze download + D3 RSF grid + B=1000 bootstraps in `--full` cells).

## 4. Strict / paper mode

```bash
export CDS_STRICT=1
```

`WAITING` / blocked gates then exit **2** (default research mode still exits 0).

## 5. Machine fingerprint

```bash
python -W default A04_machine_fingerprint.py
# → results/logs/machine_fingerprint.json  (serial/UUID redacted)
```

## Notes

- Large raw datasets / trained models under `data/` and `results/models/` are not shipped; Block B downloads, Block D trains.
- Protocol freeze: `C00_preregister_protocol.py` (content SHA from `cfg.PROTOCOL`).
- Paper SoT: Block P → `results/paper/`.
- Overrides: `CDS_MAIN_PYTHON`, `CDS_D3_PYTHON`, `CDS_REQUIREMENTS`.
- Lock file was generated on **macOS arm64 / Python 3.13.5**; other platforms should still install, but pins are author-host validated only.
