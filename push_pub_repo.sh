#!/usr/bin/env bash
# push_pub_repo.sh — Publish the *public* reproduction surface to
#   https://github.com/rafa-rodriguess/c_index_illusion_pub
#
# Include what a third party needs to *run* the pipeline (code + docs +
# dependency pins + paper builders). No data dumps, no trained models,
# no literature PDFs / paper prose dumps.
#
# Usage:
#   ./push_pub_repo.sh
#   ./push_pub_repo.sh --dry-run
#
# Auth (once):
#   gh auth login -h github.com -p https -w
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

REMOTE_URL="https://github.com/rafa-rodriguess/c_index_illusion_pub.git"
REPO_SLUG="rafa-rodriguess/c_index_illusion_pub"
REMOTE_NAME="pub"
BRANCH="main"
MAX_BYTES=$((50 * 1024 * 1024))
DRY_RUN=0
MANIFEST="$ROOT/.git_pub_manifest.txt"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

log() { printf '%s\n' "$*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    log "ERROR: required command not found: $1"
    exit 1
  }
}

need_cmd git
need_cmd find
need_cmd gh

file_size() {
  local f="$1"
  if size="$(stat -f%z "$f" 2>/dev/null)"; then
    printf '%s' "$size"
  elif size="$(stat -c%s "$f" 2>/dev/null)"; then
    printf '%s' "$size"
  else
    printf '0'
  fi
}

# ── Allowlist: paths relative to ROOT that belong on the public surface ─────
#
# Included: root pipeline .py, src/, 00_pipeline.ipynb, README/REPRODUCE,
# requirements.*, literature CODE_ACCESS stubs, results/paper builders+style,
# F01.gv, F06 script.
#
# Explicitly NOT published: PIPELINE.md, roadmap.md, artefatos.md,
# literature PDFs / paper dumps, data/, trained models, generated figures/tables.

is_allowlisted() {
  local rel="$1"

  case "$rel" in
    .venv/*|venv/*|*/__pycache__/*|*.pyc|*.pyo|.git/*) return 1 ;;
    .ipynb_checkpoints/*|*/.ipynb_checkpoints/*) return 1 ;;
    .env|.env.*|*/.env|*/.env.*) return 1 ;;
  esac

  case "$rel" in
    README.md|REPRODUCE.md) return 0 ;;
    requirements.lock|requirements.txt|.gitignore) return 0 ;;
    push_pub_repo.sh|bootstrap_envs.sh|00_pipeline.ipynb) return 0 ;;
  esac

  # Literature / access stubs (no PDFs)
  case "$rel" in
    domain1-ahmed-green-2024/CODE_ACCESS.md) return 0 ;;
    domain2-bone-winkel-reichenbach-2024/CODE_ACCESS.md) return 0 ;;
    domain3-abedi-2022/CODE_ACCESS.md|domain3-abedi-2022/AUTHOR_BINS.md) return 0 ;;
    anchor-stop-chasing-c-index/CODE_ACCESS.md) return 0 ;;
  esac

  # Paper builders (needed by P02/P03/F06) — code only, not generated artefacts
  case "$rel" in
    results/paper/builders/*.py|results/paper/style/*.py) return 0 ;;
    results/paper/figures/F01_workflow.gv) return 0 ;;
    results/paper/scripts/F06_copula_sweep_d1.py) return 0 ;;
  esac

  case "$rel" in
    *.py)
      case "$rel" in
        results/*|data/*) return 1 ;;
        src/*) return 0 ;;
        */*) return 1 ;;
        *) return 0 ;;
      esac
      ;;
  esac

  case "$rel" in
    *.ipynb)
      case "$rel" in
        data/*|results/*) return 1 ;;
        *) return 0 ;;
      esac
      ;;
  esac

  return 1
}

ensure_public_repo() {
  if ! gh auth status -h github.com >/dev/null 2>&1; then
    log "ERROR: GitHub CLI is not authenticated."
    log "  Run:  gh auth login -h github.com -p https -w"
    exit 1
  fi
  if gh repo view "$REPO_SLUG" >/dev/null 2>&1; then
    log "Remote repo exists: https://github.com/$REPO_SLUG"
  else
    log "Creating public repository https://github.com/$REPO_SLUG …"
    gh repo create "$REPO_SLUG" \
      --public \
      --description "C-index Illusion — public reproduction code (no data/results)" \
      >/dev/null
    log "Created."
  fi
}

# ── Build manifest ──────────────────────────────────────────────────────────
log "Building public file manifest…"
: > "$MANIFEST"
n_ok=0
n_skip_size=0

while IFS= read -r -d '' f; do
  rel="${f#"$ROOT"/}"
  rel="${rel#./}"
  if ! is_allowlisted "$rel"; then
    continue
  fi
  size="$(file_size "$f")"
  if (( size >= MAX_BYTES )); then
    log "  SKIP (>=50MiB): $rel"
    n_skip_size=$((n_skip_size + 1))
    continue
  fi
  printf '%s\n' "$rel" >> "$MANIFEST"
  n_ok=$((n_ok + 1))
done < <(find "$ROOT" -type f -print0 2>/dev/null)

# stable order
sort -o "$MANIFEST" "$MANIFEST"

log "  Public files to publish: $n_ok"
log "  Skipped (>=50 MiB):      $n_skip_size"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "[dry-run] Manifest:"
  sed 's/^/  - /' "$MANIFEST"
  log "[dry-run] Remote: $REMOTE_URL"
  exit 0
fi

ensure_public_repo

# ── Stage in a clean temp git repo (does not disturb local prv/main) ────────
WORKDIR="$(mktemp -d -t c_index_illusion_pub.XXXXXX)"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

log "Assembling clean tree in $WORKDIR …"
while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  dest="$WORKDIR/$rel"
  mkdir -p "$(dirname "$dest")"
  cp -p "$ROOT/$rel" "$dest"
done < "$MANIFEST"

# Minimal public README note if somehow missing (should already be copied)
if [[ ! -f "$WORKDIR/README.md" ]]; then
  printf '# C-index Illusion (public)\n\nSee REPRODUCE.md\n' > "$WORKDIR/README.md"
fi

cd "$WORKDIR"
git init -b "$BRANCH" >/dev/null
git add -A
git commit -m "Public reproduction surface $(date -u +%Y-%m-%dT%H:%MZ)" >/dev/null
git remote add origin "$REMOTE_URL"

log "Pushing to origin/${BRANCH} (force replace public allowlist tree)..."
# Public repo is an allowlisted republish; plain --force is intentional.
git push --force -u origin "$BRANCH"
cd "$ROOT"
# Record remote on the local repo for convenience (does not change checked-out files)
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
  else
    git remote add "$REMOTE_NAME" "$REMOTE_URL" 2>/dev/null || true
  fi
fi

log "Done."
log "Repo: https://github.com/$REPO_SLUG"
log "Manifest saved: $(basename "$MANIFEST") ($n_ok files)"
