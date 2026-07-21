"""
A01_setup_dirs.py — Criar estrutura de diretórios do projeto
============================================================
Cria todos os diretórios definidos em cfg.DIRS (idempotente).

Adiciona .gitkeep em diretórios vazios para o Git rastrear a árvore
sem precisar versionar dados.

Executar:
    python A01_setup_dirs.py

Critério de pronto:
    - Todos os diretórios em cfg.DIRS existem após a execução
    - Nenhum arquivo existente é removido

Notas (revisão 2026-07-13):
    - Removidos placeholders nunca usados: ``literature/``, ``notebooks/``,
      ``results/{metrics,figures}``, ``data/interim/domain3``, ``results/models/anchor``.
    - Figuras/tabelas do manuscript → ``results/paper/{figures,tables}``.
    - Métricas da escada → ``results/ladder`` (+ ``results/probes``).
    - Papers/baselines vivem em ``domain*-*/``, não em ``literature/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg

# Extra leaves not always implied as separate DIRS keys historically;
# kept explicit so a fresh clone gets the write targets the pipeline uses.
EXTRA_DIRS: list[Path] = [
    # literature_dir stubs (A00 asserts these; public repo ships CODE_ACCESS.md)
    cfg.DOMAINS["domain1"]["literature_dir"],
    cfg.DOMAINS["domain2"]["literature_dir"],
    cfg.DOMAINS["domain3"]["literature_dir"],
    cfg.ANCHOR["literature_dir"],
]


def setup_dirs() -> None:
    all_dirs = list(cfg.DIRS.values()) + EXTRA_DIRS

    print(f"{'─' * 60}")
    print("CRIANDO ESTRUTURA DE DIRETÓRIOS")
    print(f"{'─' * 60}")

    for d in sorted(set(all_dirs), key=lambda p: str(p)):
        existed = d.exists()
        d.mkdir(parents=True, exist_ok=True)

        gitkeep = d / ".gitkeep"
        # Only plant .gitkeep when the directory has no other entries
        if not any(p for p in d.iterdir() if p.name != ".gitkeep"):
            gitkeep.touch()

        rel = d.relative_to(cfg.ROOT)
        status = "já existia" if existed else "criado"
        print(f"  {status:<12}  {rel}/")

    print(f"{'─' * 60}")

    missing = [d for d in all_dirs if not d.exists()]
    if missing:
        raise RuntimeError(f"Diretórios não criados: {missing}")

    print(f"  Total: {len(set(all_dirs))} diretórios presentes.")


if __name__ == "__main__":
    setup_dirs()
    print("\nA01 concluído — estrutura de diretórios pronta.")
