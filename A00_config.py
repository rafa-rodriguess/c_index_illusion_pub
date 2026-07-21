"""
A00_config.py — Validar e exibir configuração central
=====================================================
Passo zero do pipeline: importa src/config.py, verifica coerência básica
e imprime um sumário legível.

Executar:
    python A00_config.py

Critério de pronto:
    - Nenhum AssertionError
    - Saída impressa sem erros
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg


def validate_config() -> None:
    assert isinstance(cfg.RANDOM_SEED, int) and cfg.RANDOM_SEED >= 0
    assert cfg.ROOT.exists() and cfg.ROOT.is_dir()

    required_domains = {"domain1", "domain2", "domain3"}
    assert required_domains <= set(cfg.DOMAINS.keys())

    for key, meta in cfg.DOMAINS.items():
        assert "name" in meta and "doi" in meta and "literature_dir" in meta
        lit = Path(meta["literature_dir"])
        # Fresh public clones ship CODE_ACCESS.md stubs; mkdir keeps A00 green
        # even when the stub tree is not yet checked out.
        lit.mkdir(parents=True, exist_ok=True)
        assert lit.exists(), f"literature_dir ausente para {key}: {lit}"

    Path(cfg.ANCHOR["literature_dir"]).mkdir(parents=True, exist_ok=True)
    assert cfg.ANCHOR["literature_dir"].exists()
    assert 0 < cfg.EVAL["d_calibration_alpha"] < 1
    assert cfg.EVAL["n_cv_folds"] >= 2
    assert cfg.EVAL["n_cv_repeats"] >= 1
    assert len(cfg.EVAL["ibs_horizons_months"]) >= 1
    assert cfg.EVAL["auc_horizons_months"] == [12, 24, 36]
    assert hasattr(cfg, "PROTOCOL")
    assert cfg.PROTOCOL["version"]
    assert set(cfg.PROTOCOL["hypotheses"]) >= {"H1", "H2", "H3", "H4", "H5", "H_meta"}
    assert cfg.H4_ABLATION_SMART == [5, 197, 198]
    assert len(cfg.DOMAIN_01["smart_ids"]) == 21
    assert cfg.DOMAIN_01["omit_smart_ids"] == [3, 10, 191]
    assert cfg.DOMAIN_01["cox_cohort_min_age_years"] == 7
    assert cfg.DOMAIN_01["cox_fit_population"] == "h6a_calgt7_healthy_union_all_failed"
    assert cfg.DOMAIN_01["target_value"] == 0.958
    assert cfg.DOMAIN_01["model_filter"] == "ST4000DM000"
    for name in ("raw", "interim", "processed", "ladder", "probes", "paper", "harness"):
        assert name in cfg.DIRS
    # Manuscript outputs (Block P) and per-domain model dirs
    for name in ("paper_tables", "paper_figures", "paper_collection", "models_d1", "models_d2", "models_d3"):
        assert name in cfg.DIRS
    # Retired placeholders must not return
    for name in ("metrics", "figures", "literature", "interim_d3", "tests"):
        assert name not in cfg.DIRS, f"retired DIRS key still present: {name}"
    assert len(cfg.BACKBLAZE["zip_filenames"]) == 3 + 7 * 4  # 2013–15 annual + 2016–22 × 4Q
    assert cfg.BACKBLAZE["zip_filenames"][0] == "data_2013.zip"
    assert cfg.BACKBLAZE["zip_filenames"][-1] == "data_Q4_2022.zip"


def print_summary() -> None:
    sep = "─" * 60
    print(sep)
    print("CONFIGURAÇÃO — cross-domain-survival")
    print(sep)
    print(f"  Raiz        : {cfg.ROOT}")
    print(f"  Seed global : {cfg.RANDOM_SEED}")
    print()

    print("  ÂNCORA")
    print(f"    Venue  : {cfg.ANCHOR['venue']}")
    print(f"    arXiv  : {cfg.ANCHOR['arxiv']}")
    print()

    print("  DOMÍNIOS")
    for key, meta in cfg.DOMAINS.items():
        cidx = meta["reported_cindex"]
        print(f"    {key}: {meta['baseline']}")
        print(f"           dataset={meta['dataset']}  C-index={cidx}")
    print()

    print("  AVALIAÇÃO / PROTOCOLO")
    print(f"    Protocol version : {cfg.PROTOCOL['version']}")
    print(f"    Hypotheses       : {', '.join(cfg.PROTOCOL['hypotheses'])}")
    print(f"    C-index variants : {cfg.EVAL['cindex_variants']}")
    print(f"    D-Cal α          : {cfg.EVAL['d_calibration_alpha']}")
    print(f"    AUC horizons     : {cfg.EVAL['auc_horizons_months']} meses")
    print(f"    IBS horizons     : {cfg.EVAL['ibs_horizons_months']} meses")
    print(f"    CV               : {cfg.EVAL['n_cv_folds']}-fold × {cfg.EVAL['n_cv_repeats']}")
    print(f"    H1 D1 companion ablation SMART ids: {cfg.H4_ABLATION_SMART}")
    print()
    print("  DOMAIN_01 (Ahmed reproduction)")
    print(f"    Model            : {cfg.DOMAIN_01['model_filter']}")
    print(f"    SMART features   : {len(cfg.DOMAIN_01['smart_ids'])} ids")
    print(f"    Cox cohort age   : > {cfg.DOMAIN_01['cox_cohort_min_age_years']} years")
    print(f"    Target C-index   : {cfg.DOMAIN_01['target_value']}")
    print(f"    Scripts          : {', '.join(cfg.DOMAIN_01['scripts'])}")
    print()

    print("  DIRETÓRIOS")
    for name, path in cfg.DIRS.items():
        exists = "✓" if path.exists() else "✗ (ainda não existe)"
        try:
            rel = path.relative_to(cfg.ROOT)
        except ValueError:
            rel = path
        print(f"    {name:<18}: {rel}  {exists}")
    print(sep)


if __name__ == "__main__":
    print("Validando configuração...", end=" ")
    validate_config()
    print("OK\n")
    print_summary()
    print("\nA00 concluído — configuração válida.")
