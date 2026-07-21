#!/usr/bin/env bash
# bootstrap_envs.sh — Create the two intentional interpreters for this project.
#
#   .venv/                 Main trunk (requirements.lock) — almost everything
#   conda:d3-pysurvival    Domain-3 Table-8 RSF only (PySurvival 0.1.2)
#
# Usage:
#   ./bootstrap_envs.sh           # main .venv only (default)
#   ./bootstrap_envs.sh --d3      # main + attempt Domain-3 conda env
#   ./bootstrap_envs.sh --d3-only # Domain-3 only
#
# After success:
#   source .venv/bin/activate
#   export CDS_D3_PYTHON="$(conda info --base)/envs/d3-pysurvival/bin/python"
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DO_MAIN=1
DO_D3=0

case "${1:-}" in
  --d3) DO_D3=1 ;;
  --d3-only) DO_MAIN=0; DO_D3=1 ;;
  ""|--main) ;;
  -h|--help)
    sed -n '2,18p' "$0"
    exit 0
    ;;
  *)
    echo "Unknown option: $1 (try --help)" >&2
    exit 1
    ;;
esac

log() { printf '%s\n' "$*"; }

# ── Main trunk ──────────────────────────────────────────────────────────────
if [[ "$DO_MAIN" -eq 1 ]]; then
  log "══ Main trunk (.venv) ══"
  if [[ ! -x "$ROOT/.venv/bin/python" && ! -x "$ROOT/.venv/Scripts/python.exe" ]]; then
    PY="${CDS_BOOTSTRAP_PYTHON:-python3}"
    command -v "$PY" >/dev/null 2>&1 || PY=python
    log "  Creating .venv with: $PY"
    "$PY" -m venv "$ROOT/.venv"
  else
    log "  .venv already exists"
  fi
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    VENV_PY="$ROOT/.venv/bin/python"
  else
    VENV_PY="$ROOT/.venv/Scripts/python.exe"
  fi
  "$VENV_PY" -m pip install -U pip
  "$VENV_PY" -m pip install -r "$ROOT/requirements.lock"
  log "  OK — activate with:  source .venv/bin/activate"
  log "  Interpreter: $VENV_PY"
fi

# ── Domain-3 PySurvival ─────────────────────────────────────────────────────
if [[ "$DO_D3" -eq 1 ]]; then
  log ""
  log "══ Domain-3 (conda env d3-pysurvival) ══"
  if ! command -v conda >/dev/null 2>&1; then
    log "ERROR: conda not found on PATH."
    log "  Install Miniconda/Anaconda, then re-run: ./bootstrap_envs.sh --d3"
    log "  Manual setup: domain3-abedi-2022/CODE_ACCESS.md"
    exit 1
  fi

  if conda env list | awk '{print $1}' | grep -qx 'd3-pysurvival'; then
    log "  conda env d3-pysurvival already exists"
  else
    log "  Creating conda env d3-pysurvival (python=3.9)…"
    conda create -y -n d3-pysurvival python=3.9
  fi

  # shellcheck disable=SC1091
  # Prefer conda run so we do not pollute the caller shell
  set +e
  conda run -n d3-pysurvival python -c "import pysurvival" >/dev/null 2>&1
  HAS_PS=$?
  set -e
  if [[ "$HAS_PS" -ne 0 ]]; then
    log "  Installing PySurvival 0.1.2 into d3-pysurvival…"
    log "  (If the build fails on tp_print, apply the one-line C++ patch — see CODE_ACCESS.md)"
    set +e
    conda run -n d3-pysurvival pip install "numpy<1.24" "pandas<2" "scikit-learn==0.22.2" scipy pyarrow
    conda run -n d3-pysurvival pip install pysurvival==0.1.2
    PS_RC=$?
    set -e
    if [[ "$PS_RC" -ne 0 ]]; then
      log "WARNING: pysurvival install failed (often the macOS tp_print ABI issue)."
      log "  Follow the patch steps in domain3-abedi-2022/CODE_ACCESS.md, then:"
      log "    conda run -n d3-pysurvival pip install pysurvival==0.1.2"
      exit 1
    fi
  else
    log "  pysurvival already importable"
  fi

  CONDA_BASE="$(conda info --base)"
  D3_PY="$CONDA_BASE/envs/d3-pysurvival/bin/python"
  if [[ ! -x "$D3_PY" ]]; then
    D3_PY="$CONDA_BASE/envs/d3-pysurvival/python.exe"
  fi
  log "  OK — export CDS_D3_PYTHON=\"$D3_PY\""
  log "  (00_pipeline.ipynb resolves this path automatically when present)"
fi

log ""
log "Done. Two-env layout:"
log "  main  → .venv                  (A–H, P, D02b, …)"
log "  d3    → d3-pysurvival          (D02_DOMAIN_03_train_rsf.py only)"
