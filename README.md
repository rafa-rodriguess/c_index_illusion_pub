# Cross-Domain Survival Evaluation

Scientific multi-domain survival audit (*C-index Illusion*), anchored on Lillelund et al. (ICML 2026).

## Quick start

See **[REPRODUCE.md](REPRODUCE.md)** for clone → two envs → run (Kaggle credentials required for Bondora).

```bash
git clone https://github.com/rafa-rodriguess/c_index_illusion_pub.git
cd c_index_illusion_pub
./bootstrap_envs.sh          # .venv + requirements.lock
source .venv/bin/activate
python -W default A00_config.py && python -W default A01_setup_dirs.py
python -W default A03_env_check.py
# Optional Domain-3 RSF: ./bootstrap_envs.sh --d3
```

Full pipeline: open `00_pipeline.ipynb` with the **`.venv` kernel** and run top-to-bottom.

## Layout

| Path | Role |
|------|------|
| `src/config.py` | Central config + seeds |
| `00_pipeline.ipynb` | Pipeline orchestrator (canonical) |
| `results/paper/` | Manuscript builders / figures SoT (generated numbers under gitignore) |
| `requirements.lock` | Exact pins for reproduction |
| `domain*-*/CODE_ACCESS.md` | Baseline code / data pointers |

## Strict mode

`CDS_STRICT=1` makes `WAITING` gates fail (exit 2) in pipeline scripts that honour it.
