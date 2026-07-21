"""
config.py — Configuração central do projeto
============================================
Importar em todos os scripts de pipeline:

    from src.config import cfg

Parâmetros de análise vivem aqui. Scripts de etapa não hard-codam
escolhas metodológicas — leem de `cfg`.
"""

from pathlib import Path

# ── Raiz do projeto ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# ── Seed global (fixo para toda análise) ─────────────────────────────────────
RANDOM_SEED = 42

# ── Estrutura de diretórios ───────────────────────────────────────────────────
DIRS = {
    "raw":              ROOT / "data" / "raw",
    "raw_backblaze":    ROOT / "data" / "raw" / "backblaze",
    "raw_bondora":      ROOT / "data" / "raw" / "bondora",
    "raw_stackexchange": ROOT / "data" / "raw" / "stackexchange",
    "raw_anchor":       ROOT / "data" / "raw" / "anchor",
    "interim":          ROOT / "data" / "interim",
    "interim_d1":       ROOT / "data" / "interim" / "domain1",
    "interim_d2":       ROOT / "data" / "interim" / "domain2",
    "processed":        ROOT / "data" / "processed",
    "processed_d1":     ROOT / "data" / "processed" / "domain1",
    "processed_d2":     ROOT / "data" / "processed" / "domain2",
    "processed_d3":     ROOT / "data" / "processed" / "domain3",
    "processed_anchor": ROOT / "data" / "processed" / "anchor",
    "models":           ROOT / "results" / "models",
    "models_d1":        ROOT / "results" / "models" / "domain1",
    "models_d2":        ROOT / "results" / "models" / "domain2",
    "models_d3":        ROOT / "results" / "models" / "domain3",
    "ladder":           ROOT / "results" / "ladder",
    "probes":           ROOT / "results" / "probes",
    "logs":             ROOT / "results" / "logs",
    "reproduction":     ROOT / "results" / "reproduction",
    "harness":          ROOT / "results" / "harness",
    "paper":            ROOT / "results" / "paper",
    "paper_tables":     ROOT / "results" / "paper" / "tables",
    "paper_figures":    ROOT / "results" / "paper" / "figures",
    "paper_appendix":   ROOT / "results" / "paper" / "appendix",
    "paper_collection": ROOT / "results" / "paper" / "collection",
    "src":              ROOT / "src",
}

# ── Domínios / baselines ─────────────────────────────────────────────────────
DOMAINS = {
    "domain1": {
        "name": "engineering_reliability",
        "baseline": "Ahmed & Green, 2024",
        "doi": "10.1007/s00521-024-10479-6",
        "dataset": "Backblaze Drive Stats",
        "event": "hard_drive_failure",
        "reported_cindex": 0.958,
        "literature_dir": ROOT / "domain1-ahmed-green-2024",
    },
    "domain2": {
        "name": "p2p_credit",
        "baseline": "Bone-Winkel & Reichenbach, 2024",
        "doi": "10.1007/s42521-024-00114-3",
        "dataset": "Bondora P2P Loans",
        "event": "loan_default",
        "reported_cindex": None,  # rating-based evaluation
        "literature_dir": ROOT / "domain2-bone-winkel-reichenbach-2024",
    },
    "domain3": {
        "name": "digital_platforms",
        "baseline": "Abedi Firouzjaei, 2022",
        "doi": "10.1007/s13278-022-00914-8",
        "dataset": "Stack Exchange (Pol / DS / CS)",
        "event": "user_disengagement",
        "reported_cindex": (0.66, 0.76),
        "literature_dir": ROOT / "domain3-abedi-2022",
        "github": "https://github.com/habedi/SurvivalAnalysisQACommunities",
    },
}

# ── Fontes de download (URLs físicas) ────────────────────────────────────────
DATA_URLS = {
    "backblaze_index": (
        "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data"
    ),
    "backblaze_zip_template": (
        "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/{filename}"
    ),
    "bondora_loan_xlsx": (
        "https://sabanners001.blob.core.windows.net/statistics/public/"
        "loan_dataset_investor.xlsx"
    ),
    "bondora_stats_page": "https://bondora.com/en/public-statistics/",
    # Canonical DOMAIN_02 LoanData (Kaggle mirror of historical ~100-col schema).
    "bondora_loandata_kaggle": (
        "https://www.kaggle.com/api/v1/datasets/download/marcobeyer/bondora-p2p-loans"
    ),
    "bondora_loandata_kaggle_slug": "marcobeyer/bondora-p2p-loans",
    # Historical names cited by Bone-Winkel & Reichenbach (retrieved 2024-01-03).
    # Post Go&Grow rebrand these return 403/404; kept for documentation only.
    "bondora_loan_zip_legacy": "https://www.bondora.com/marketing/media/LoanData.zip",
    "bondora_repayments_zip_legacy": (
        "https://www.bondora.com/marketing/media/RepaymentsData.zip"
    ),
    "stackexchange_archive": "https://archive.org/download/stackexchange/",
    "stackexchange_politics": (
        "https://archive.org/download/stackexchange/politics.stackexchange.com.7z"
    ),
    "stackexchange_datascience": (
        "https://archive.org/download/stackexchange/datascience.stackexchange.com.7z"
    ),
    "stackexchange_cs": (
        "https://archive.org/download/stackexchange/cs.stackexchange.com.7z"
    ),
    "abedi_processed_cs": (
        "https://media.githubusercontent.com/media/habedi/"
        "SurvivalAnalysisQACommunities/main/processed_data/cs/user_features.csv"
    ),
    "abedi_processed_ds": (
        "https://media.githubusercontent.com/media/habedi/"
        "SurvivalAnalysisQACommunities/main/processed_data/ds/user_features.csv"
    ),
    "abedi_processed_p": (
        "https://media.githubusercontent.com/media/habedi/"
        "SurvivalAnalysisQACommunities/main/processed_data/p/user_features.csv"
    ),
}

# ── Âncora metodológica ──────────────────────────────────────────────────────
ANCHOR = {
    "title": "Position: Stop Chasing the C-index when Evaluating Survival Analysis Models",
    "venue": "ICML 2026, Spotlight",
    "arxiv": "2506.02075",
    "doi": "10.48550/arXiv.2506.02075",
    "code": "https://github.com/thecml/position-cindex",
    "code_repo": "thecml/position-cindex",
    "code_ref": "main",
    "literature_dir": ROOT / "anchor-stop-chasing-c-index",
    "raw_dir": ROOT / "data" / "raw" / "anchor" / "position-cindex",
    "processed_dir": ROOT / "data" / "processed" / "anchor",
    "results_dir": ROOT / "results" / "reproduction",
    # Harness-check role (not Domain 04): validate our ladder APIs vs author synthetic expt
    "role": "harness_check",
    # Exact data_cfg from author ladder_hypo.ipynb
    "data_cfg": {
        "alpha_e1": 12,
        "gamma_e1": 3,  # censoring
        "alpha_e2": 17,
        "gamma_e2": 4,  # event
        "n_samples": 10000,
        "n_features": 10,
    },
    "n_seeds_default": 100,  # paper Table 2 / Figure 5
    "n_seeds_smoke": 5,  # fast plumbing check
    "author_files": [
        "dgp.py",
        "utility.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
        "ladder_hypo.ipynb",
        "plot_metrics.ipynb",
        "stats.ipynb",
        "data/metrics_used.csv",
        "data/correct_evaluation.csv",
        "data/references.bib",
        "data/.gitkeep",
    ],
    # Protocol from paper §5.1 / Table 2 (oracle targets to beat)
    "n_seeds": 100,
    "model": "CoxPH",
    "copula": "clayton",
    "scenarios": [
        {"id": "random", "label": "Random", "k_tau": 0.0, "copula": None},
        {"id": "independent", "label": "Independent", "k_tau": 0.0, "copula": None},
        {"id": "dep_tau25", "label": "Dependent (τ=0.25)", "k_tau": 0.25, "copula": "clayton"},
        {"id": "dep_tau50", "label": "Dependent (τ=0.50)", "k_tau": 0.50, "copula": "clayton"},
        {"id": "dep_tau75", "label": "Dependent (τ=0.75)", "k_tau": 0.75, "copula": "clayton"},
    ],
    # Table 2 — paper oracle means (primary numeric targets for AN03)
    "table2_oracle": {
        "random": {
            "n_events": 2641,
            "censor_pct": 73.6,
            "ci_oracle_mean": 0.634,
            "ci_oracle_sd": 0.018,
            "ibs_oracle_mean": 0.090,
            "ibs_oracle_sd": 0.040,
        },
        "independent": {
            "n_events": 3157,
            "censor_pct": 68.4,
            "ci_oracle_mean": 0.634,
            "ci_oracle_sd": 0.018,
            "ibs_oracle_mean": 0.084,
            "ibs_oracle_sd": 0.037,
        },
        "dep_tau25": {
            "n_events": 2969,
            "censor_pct": 70.3,
            "ci_oracle_mean": 0.628,
            "ci_oracle_sd": 0.021,
            "ibs_oracle_mean": 0.132,
            "ibs_oracle_sd": 0.096,
        },
        "dep_tau50": {
            "n_events": 2758,
            "censor_pct": 72.4,
            "ci_oracle_mean": 0.618,
            "ci_oracle_sd": 0.025,
            "ibs_oracle_mean": 0.199,
            "ibs_oracle_sd": 0.144,
        },
        "dep_tau75": {
            "n_events": 2536,
            "censor_pct": 74.6,
            "ci_oracle_mean": 0.609,
            "ci_oracle_sd": 0.030,
            "ibs_oracle_mean": 0.245,
            "ibs_oracle_sd": 0.157,
        },
    },
}

# ── Avaliação / pré-registro (fonte para C00 freeze) ─────────────────────────
EVAL = {
    "alpha": 0.05,
    "n_bootstrap": 1000,
    "n_permutation": 10_000,
    "cindex_variants": ["harrell", "antolini", "uno"],
    "d_calibration_alpha": 0.05,
    "d_calibration_bins": 10,
    "auc_horizons_months": [12, 24, 36],  # H5 — Uno time-dependent AUC
    "ibs_horizons_months": [12, 24, 36],  # ladder proper scores (not H5 decision)
    "n_cv_folds": 5,
    "n_cv_repeats": 30,  # Domínio 3 (Abedi)
}

# H4: broad SMART ablation survey (leave-one-out + reported hits) — existential.
# Feature drop used only to build ``cox_ablated`` for H1 D1 ranking (not an H4 claim):
H4_ABLATION_SMART = [5, 197, 198]

# Frozen hypothesis protocol (roadmap §C00 + §8) — authoritative for C00 script
PROTOCOL = {
    "version": "2026-07-12.c00.v5.1",
    "roadmap_sections": ["C00", "8", "8.1", "8.2"],
    "amendments": [
        {
            "id": "C00.1_subject_bootstrap",
            "date": "2026-07-12",
            "from_version": "2026-07-12.c00.v4",
            "to_version": "2026-07-12.c00.v5.1",
            "reason": (
                "Rank-permutation p is unreachable for k<=3 models "
                "(min unsmoothed p = 1/k!). Formal gate = subject-level "
                "stratified bootstrap p(tau_K=1) with tau_obs<=0.5; "
                "rank-permutation retained as sensitivity only. "
                "v5.1 clarifies p = (1+#{tau_b>=1})/(B+1) "
                "(percentile CI alone is too coarse for k=2 discrete tau)."
            ),
        },
    ],
    "freeze_decisions": {
        "C00.1": {
            "id": "H1_decision_rule",
            "official": (
                "Reject H0 in a domain iff observed tau_K <= 0.5 AND the "
                "subject-level stratified bootstrap (B=1000) p-value for "
                "H0: tau_K = 1 is < alpha, with "
                "p = (1 + #{tau_b >= 1}) / (B + 1). "
                "Percentile CI for tau_K is reported; rank-permutation is "
                "sensitivity-only (unreachable for k<=3)."
            ),
            "binary_inversion_cross_domain": "descriptive_only",
            "primary_test": "subject_level_bootstrap_tau_K",
            "sensitivity_test": "rank_permutation",
        },
        "C00.2": {
            "id": "H1_ranking_objects",
            "rankings": {
                "domain1": ["cox_full", "cox_ablated"],
                "domain2": ["cox_classical", "cox_xgboost"],
                "domain3": ["rsf_behavioural", "rsf_content", "rsf_combined"],
            },
        },
        "C00.3": {
            "id": "multiple_testing_family",
            "scheme": "two_level_holm",
            "inner": (
                "Holm within hypothesis to collapse to one decision p "
                "(H1: domains; H3: rating strata; H2/H4/H5: already single)."
            ),
            "outer": "Holm-Bonferroni over the 5 primary hypothesis p-values",
            "meta_family_size": 5,
        },
    },
    "globals": {
        "alpha": 0.05,
        "n_bootstrap": 1000,
        "n_permutation": 10_000,
        "random_seed": RANDOM_SEED,
    },
    "hypotheses": {
        "H1": {
            "title": "Ranking inversion under metric-assumption misalignment",
            "H0": "Within each domain, C-index ranking equals IPCW-IBS ranking (tau_K = 1).",
            "H1": "tau_K < 1 in at least one domain (math alternative; decision = C00.1).",
            "statistic": (
                "Kendall tau_K (C-index rank vs IPCW-IBS rank); "
                "subject-level stratified bootstrap CI (primary); "
                "rank permutation (sensitivity)."
            ),
            "decision": "C00.1",
            "domains": ["domain1", "domain2", "domain3"],
        },
        "H2": {
            "title": "Discrimination–reliability dissociation",
            "H0": "Every model with C_H >= 0.90 passes D-Calibration (p >= 0.05).",
            "H1": "Exists model with C_H >= 0.90 and p_D-cal < 0.05.",
            "statistic": "D-Calibration chi2, 10 quantile bins (Haider et al., 2020)",
            "decision": (
                "Reject if Backblaze reproduction with C_H>=0.90 has p_D-cal < 0.05."
            ),
            "domains": ["domain1"],
            "c_h_threshold": 0.90,
        },
        "H3": {
            "title": "Directional bias from competing-risks blindness",
            "H0": (
                "|F_naive_default(12m) - F_AJ_default(12m)| <= 0.02 "
                "and bootstrap CI contains 0."
            ),
            "H1": (
                "Absolute difference > 0.02, CI excludes 0; "
                "predicted direction F_naive > F_AJ."
            ),
            "statistic": "Absolute CIF difference at 12 months; B=1000 percentile CI",
            "decision": (
                "Reject if delta > 0.02 AND positive sign in >= 3 of 5 rating strata."
            ),
            "domains": ["domain2"],
            "delta": 0.02,
            "horizon_months": 12,
            "n_rating_strata": 5,
            "min_strata_consistent": 3,
        },
        "H4": {
            "title": "Discrimination inflated by a small SMART subset (broad ablation)",
            "H0": (
                "No SMART ablation set yields Delta C_H >= 0.03 with "
                "non-overlapping paired bootstrap CIs (full vs ablated)."
            ),
            "H1": (
                "There exists a SMART ablation set with Delta C_H >= 0.03 "
                "and non-overlapping CIs — i.e. discrimination is concentrated "
                "in a small feature subset (structural leakage / near-failure signal)."
            ),
            "statistic": (
                "Broad ablation survey (leave-one-out + reported subsets); "
                "paired bootstrap (B=1000) on Delta C_H for sets that clear the floor"
            ),
            "decision": (
                "Reject H0 if at least one ablation set meets Delta >= 0.03 "
                "with non-overlapping full vs ablated CIs."
            ),
            "domains": ["domain1"],
            "delta_c_threshold": 0.03,
            "method": "broad_ablation_survey",
            "honesty_note": (
                "Hypothesis is existential over ablation sets (LOO survey + "
                "paired bootstrap on floor hits), not a claim that any fixed "
                "SMART ID list was known a priori. Primary exhibits: "
                "F00_sens1_leave_one_out and H04_paired_bootstrap_loo."
            ),
        },
        "H5": {
            "title": (
                "Horizon-specific proper-score degradation masked by global C-index"
            ),
            "H0": (
                "IPCW Brier score is non-increasing (or flat) across "
                "t in {12,24,36} months, or any rise is within 2 combined SEs."
            ),
            "H1": (
                "Brier(t) increases monotonically on {12,24,36}; "
                "Brier(36)-Brier(12) exceeds 2 combined bootstrap SEs; "
                "while global Harrell C remains in the reported band [0.66, 0.76]."
            ),
            "statistic": (
                "Pointwise IPCW Brier at month horizons (sksurv) + bootstrap SE; "
                "time-dependent AUC retained as sensitivity appendix"
            ),
            "decision": (
                "Reject if (for at least one frozen D3 RSF) Brier mono-increases, "
                "Delta Brier(36-12) > 2 SE, AND global C in [0.66, 0.76]."
            ),
            "domains": ["domain3"],
            "horizons_months": [12, 24, 36],
            "global_cindex_band": [0.66, 0.76],
            "primary_metric": "ipcw_brier_at_horizons",
            "sensitivity_metric": "cumulative_dynamic_auc",
            "method": "brier_horizon_degradation",
            "honesty_note": (
                "Protocol primary was first written as Uno AUC(t) decay; AUC strip "
                "did not reject. Fase C roadmap already named Brier@θ as the D3 "
                "exhibit. H5 primary is Brier degradation under stable C; AUC "
                "remains a reported sensitivity that did not close."
            ),
        },
        "H_meta": {
            "title": "C-index alone is not a reliable proxy outside healthcare",
            "H0": (
                "C-index-only evaluation is a reliable proxy for "
                "assumption-aligned evaluation cross-domain outside healthcare."
            ),
            "H1": (
                "Rejected if >= 3 of 5 primary hypotheses (H1-H5) are rejected "
                "under the outer Holm family of 5 (C00.3)."
            ),
            "min_rejected_of_five": 3,
        },
    },
    "reproduction_targets": {
        "domain1": {"metric": "harrell_cindex", "reported": 0.958},
        "domain2": {
            "metric": "rating_stratification",
            "reported": "cox_classical + cox_xgboost vs Bondora ratings",
        },
        "domain3": {
            "metric": "rsf_cindex_band",
            "reported_min": 0.66,
            "reported_max": 0.76,
            "protocol": "5-fold x 30 runs",
        },
    },
}

# ── Backblaze (Domain 1) — Ahmed & Green Cox window 2013–2022 ────────────────
# Annual zips for 2013–2015; quarterly zips from 2016 onward.
_BACKBLAZE_ZIPS: list[str] = [
    "data_2013.zip",
    "data_2014.zip",
    "data_2015.zip",
]
for _year in range(2016, 2023):
    for _q in range(1, 5):
        _BACKBLAZE_ZIPS.append(f"data_Q{_q}_{_year}.zip")

BACKBLAZE = {
    "model_filter": "ST4000DM000",  # Seagate model used in Ahmed & Green
    "period_start": "2013-01-01",
    "period_end": "2022-12-31",
    "zip_filenames": _BACKBLAZE_ZIPS,
    "timeout_s": 120,
    "chunk_size": 1 << 20,  # 1 MiB
}

# ── Domain 1 reproduction protocol (Ahmed & Green 2024) ──────────────────────
# Source: s00521-024-10479-6 §§4–7.1. Split/cohort live here — NOT in C01.
_DOMAIN_01_SMART = [
    1, 4, 5, 7, 9, 12, 183, 184, 187, 188, 189, 190, 192, 193, 194,
    197, 198, 199, 240, 241, 242,
]  # V1–V21; SMART 3, 10, 191 omitted (constant zero)

DOMAIN_01 = {
    "key": "domain1",
    "baseline": "Ahmed & Green, 2024",
    "doi": "10.1007/s00521-024-10479-6",
    "model_filter": BACKBLAZE["model_filter"],
    "period_start": BACKBLAZE["period_start"],
    "period_end": BACKBLAZE["period_end"],
    "n_drives_reported": 37_037,
    "smart_ids": _DOMAIN_01_SMART,
    "omit_smart_ids": [3, 10, 191],
    "smart_value_pref": "raw",  # raw / normalized / worst — paper uses sensor raw for analysis
    "cox_cohort_min_age_years": 7,
    "cox_cohort_reported": {"healthy": 12_993, "failed": 4_889},
    "failure_horizon_days": 15,  # DL labelling; Cox GOF uses survival times — document in D00/D02
    "dl_split": {
        "test_frac": 0.20,
        "val_frac_of_train": 0.10,
        "scaling": "max_abs_fit_on_train_only",
    },
    "cox_backend": "lifelines.CoxPHFitter",
    # H6a (smoke 2026-07-12): in-sample C≈0.9595 vs paper 0.958.
    # Healthy with calendar span (last−first)/365.25 > 7y UNION all failed.
    # Paper §4.1 counts read as informative sampling of older healthy + failures.
    "cox_fit_population": "h6a_calgt7_healthy_union_all_failed",
    "cox_cohort_age_basis": "calendar_span_years",
    # Small L2 needed for lifelines convergence (smart_190≡smart_194 collinearity).
    # Paper does not state a penalizer; documented as deviation in D02/D03.
    "cox_penalizer": 0.01,
    "cox_drop_collinear_smart": [194],  # identical to SMART 190 on Seagate ST4000DM000
    "target_metric": "harrell_cindex",
    "target_value": 0.958,
    "author_code_url": "https://gitlab.com/Jishan/deeplearning2023",
    # Verified 2026-07-12: logged-in browser still 404 (deleted/moved/inaccessible).
    "author_code_status": "url_404_unavailable",
    "scripts": [
        "D00_DOMAIN_01_features_backblaze.py",
        "D01_DOMAIN_01_split_ahmed.py",
        "D02_DOMAIN_01_train_cox.py",
        "D03_DOMAIN_01_gap.py",
    ],
}

# Placeholders for parallel lanes (filled when those threads start)
DOMAIN_02 = {
    "key": "domain2",
    "baseline": "Bone-Winkel & Reichenbach, 2024",
    "doi": "10.1007/s42521-024-00114-3",
    "loan_duration_months": 36,  # paper focuses on 3-year loans
    # Paper protocol (retrieved 2024-01-03).
    "paper_retrieve_date": "2024-01-03",
    "loan_date_min": "2014-01-01",
    "test_year_end": "2020-12-31",  # paper §3.3: test = 2020 originations only
    "temporal_split_date_paper": "2020-01-01",
    "temporal_split_date": "2020-01-01",
    "temporal_split_date_adapted": "2017-01-01",  # legacy fallback (Wayback 2018)
    "train_val_pre_split": {"train_frac": 0.90, "val_frac": 0.10},
    # Paper Table 1 / Appendix D: same count as Bondora (AA–F = 7), equal-mass on train HR
    "n_rating_strata": 7,
    "rating_labels": ["AA", "A", "B", "C", "D", "E", "F"],
    # Table 1 / footnote 14: Dömötör completed = Bondora-closed OR ≥1y without payment
    "completed_inactive_days": 365,
    "cox_backend": "lifelines.CoxPHFitter",
    "xgb_backend": "xgboost.XGBRegressor objective=survival:cox",
    "repayments_required_for_irr": True,
    "paper_table1_boosted_aa_default_rate": 0.1445,
    "paper_table1_bondora_aa_default_rate": 0.1726,
    "paper_table1_boosted_aa_irr": 0.1563,
    "paper_table1_bondora_aa_irr": -0.0320,
    # Direct model discrimination (§4.1 / footnote 13)
    "paper_cindex_linear_test": 0.659,
    "paper_cindex_boosted_test": 0.674,
    # XGB-Cox HPO — bounds taken from paper where stated (§3.4, §3.7, App. B):
    #   "up to 10 layers deep and … up to 2594 trees"; Optuna TPE on validation;
    #   GPU XGBoost (fallback: CPU hist — build has no CUDA).
    # Unspecified in paper (eta, subsample, reg, trial count): standard ranges + larger budget.
    "optuna_trials": 80,
    "xgb_hpo": {
        "max_depth": [1, 10],  # paper §3.7 / App B
        "num_boost_round_max": 2594,  # paper §3.7 / App B
        "early_stopping_rounds": 75,  # val selection (paper: optimize HPs on validation)
        "eta": [0.005, 0.2],  # not in paper; wider low end for deep ensembles
        "subsample": [0.5, 1.0],
        "colsample_bytree": [0.5, 1.0],
        "min_child_weight": [1.0, 20.0],
        "lambda": [1e-3, 10.0],
        "alpha": [1e-3, 10.0],
        "prefer_gpu": True,
    },
    "data_source_preferred": "LoanData.csv",  # Kaggle marcobeyer dump
    "data_source_url": (
        "https://www.kaggle.com/api/v1/datasets/download/marcobeyer/bondora-p2p-loans"
    ),
    "author_code_status": "not_found",
    "scripts": [
        "D00_DOMAIN_02_align_loan_vintage.py",
        "D01_DOMAIN_02_features_bondora.py",
        "D02_DOMAIN_02_split_temporal.py",
        "D03_DOMAIN_02_train_cox_xgb.py",
        "D04_DOMAIN_02_gap.py",
    ],
}

DOMAIN_03 = {
    "key": "domain3",
    "baseline": "Abedi Firouzjaei, 2022",
    "doi": "10.1007/s13278-022-00914-8",
    "github": "https://github.com/habedi/SurvivalAnalysisQACommunities",
    "period_end": "2021-05",
    "communities": ("p", "ds", "cs"),  # Politics, Data Science, Computer Science
    "community_labels": {"p": "Pol", "ds": "DS", "cs": "CS"},
    "theta_months": (24, 36),
    "feature_sets": {
        "behavioural": [
            "CastDownVotes",
            "CastUpVotes",
            "QuestionCount",
            "AnswerCount",
            "CommentCount",
        ],
        "content": [
            "AvgQuestionViewCount",
            "AvgQuestionCommentCount",
            "AvgQuestionScore",
            "AvgAnswerScore",
            "AvgAnswerCommentCount",
            "AvgCommentScore",
        ],
        "combined": None,  # behavioural + content at runtime
    },
    "duration_col": "MonthsActive",
    "inactivity_col": "MonthsSinceLastActivity",
    "event_col": "disengaged",
    # Paper §5.2: train/eval only on Q ∪ A ∪ C ∪ U ∪ D (Table 6).
    # Operationalized as any of these counts > 0 (author notebook had no filter).
    "contributor_filter": True,
    "contributor_cols": (
        "QuestionCount",
        "AnswerCount",
        "CommentCount",
        "CastUpVotes",
        "CastDownVotes",
    ),
    # Author notebook (survival analysis using RSF.ipynb) + paper PySurvival
    "rsf": {
        "backend": "pysurvival.models.survival_forest.RandomSurvivalForestModel",
        "author_backend": "pysurvival.RandomSurvivalForestModel",
        "n_estimators": 5,
        "max_depth": 5,
        "min_samples_leaf": 30,
        "max_features": "sqrt",
        "sample_size_pct": 0.63,
        "importance_mode": "permutation",
        "n_jobs": -1,
    },
    # Author notebook: train_test_split(test_size=0.01) then KFold(n_splits=5) without shuffle
    "cv": {
        "n_folds": 5,
        "n_runs": 30,
        "holdout_frac": 0.01,
        "kfold_shuffle": False,
    },
    "target_cindex_band": [0.66, 0.76],
    # Table 8 mean C-index (paper)
    "paper_table8": {
        ("p", "behavioural", 24): 0.75,
        ("p", "behavioural", 36): 0.76,
        ("p", "content", 24): 0.68,
        ("p", "content", 36): 0.68,
        ("p", "combined", 24): 0.75,
        ("p", "combined", 36): 0.76,
        ("ds", "behavioural", 24): 0.66,
        ("ds", "behavioural", 36): 0.66,
        ("ds", "content", 24): 0.61,
        ("ds", "content", 36): 0.63,
        ("ds", "combined", 24): 0.68,
        ("ds", "combined", 36): 0.70,
        ("cs", "behavioural", 24): 0.68,
        ("cs", "behavioural", 36): 0.68,
        ("cs", "content", 24): 0.62,
        ("cs", "content", 36): 0.63,
        ("cs", "combined", 24): 0.69,
        ("cs", "combined", 36): 0.68,
    },
    "scripts": [
        "D00_DOMAIN_03_features_stackexchange.py",
        "D01_DOMAIN_03_cv_protocol.py",
        "D02_DOMAIN_03_train_rsf.py",
        "D03_DOMAIN_03_gap.py",
    ],
}

# ── Bondora (Domain 2) — Bone-Winkel & Reichenbach ───────────────────────────
# Paper (Bondora, 2023a): loan dataset + repayments dataset, retrieved 2024-01-03.
# Current public portal exposes the loan file as xlsx only; repayments is not
# offered at a stable public URL after the Go&Grow rebrand.
BONDORA = {
    "paper_retrieved": "2024-01-03",
    # Canonical DOMAIN_02 input (Kaggle marcobeyer/bondora-p2p-loans)
    "loandata_local_name": "LoanData.csv",
    "loandata_sha256": (
        "bbfd804f442fb3a086728a0e98bf4dc87c00494a58785ced1b0c98801923cb60"
    ),
    "loandata_bytes": 229_669_536,
    # Optional legacy portal file (not used by DOMAIN_02 Cox)
    "loan_local_name": "loan_dataset_investor.xlsx",
    "repayments_local_name": "repayments_dataset.csv",
    "repayments_required_for": ["IRR", "rating_return_tables"],
    "timeout_s": 600,
    "chunk_size": 1 << 20,
}

# ── Stack Exchange (Domain 3) — Abedi Firouzjaei 2022 ────────────────────────
# Prefer the author's GitHub artefacts (preprocessed features used in experiments).
# Raw community tables are also shipped in the same repo for auditability.
# Archive.org dumps remain an optional upstream source (not required for B02).
_ABEDI_BASE = (
    "https://media.githubusercontent.com/media/habedi/"
    "SurvivalAnalysisQACommunities/main"
)
STACKEXCHANGE = {
    "github_repo": "https://github.com/habedi/SurvivalAnalysisQACommunities",
    "communities": ("p", "ds", "cs"),  # Politics, Data Science, Computer Science
    "period_end": "2021-05",  # paper: inception → May 2021
    "timeout_s": 300,
    "chunk_size": 1 << 20,
    "files": [
        # Preprocessed URVs used by RSF (primary reproduction target)
        "processed_data/p/user_features.csv",
        "processed_data/ds/user_features.csv",
        "processed_data/cs/user_features.csv",
        # Author-provided raw extracts (same GitHub release as the paper)
        "raw_data/p/users.csv.gz",
        "raw_data/p/questions.csv.gz",
        "raw_data/p/answers.csv.gz",
        "raw_data/p/comments.csv.gz",
        "raw_data/ds/users.csv.gz",
        "raw_data/ds/questions.csv.gz",
        "raw_data/ds/answers.csv.gz",
        "raw_data/ds/comments.csv.gz",
        "raw_data/cs/users.csv.gz",
        "raw_data/cs/questions.csv.gz",
        "raw_data/cs/answers.csv.gz",
        "raw_data/cs/comments.csv.gz",
    ],
    "url_template": _ABEDI_BASE + "/{path}",
}


class _Cfg:
    """Namespace for `from src.config import cfg`."""

    ROOT = ROOT
    RANDOM_SEED = RANDOM_SEED
    DIRS = DIRS
    DOMAINS = DOMAINS
    DATA_URLS = DATA_URLS
    ANCHOR = ANCHOR
    EVAL = EVAL
    PROTOCOL = PROTOCOL
    H4_ABLATION_SMART = H4_ABLATION_SMART
    BACKBLAZE = BACKBLAZE
    DOMAIN_01 = DOMAIN_01
    DOMAIN_02 = DOMAIN_02
    DOMAIN_03 = DOMAIN_03
    BONDORA = BONDORA
    STACKEXCHANGE = STACKEXCHANGE


cfg = _Cfg()
