# Domain 2 — external code / documents

## Author code

- Paper (Bone-Winkel & Reichenbach, 2024) cites software (lifelines, XGBoost, Optuna, tvm/xirr) but **no public analysis repository** was found (2026-07-11).
- Zenodo link in the PDF (`10.5281/zenodo.7883870`) is a **lifelines release**, not author code.
- Implication: reproduce from the published text (§3–4 + appendices).

## Datasets

| Dataset | Paper use | Status |
|---------|-----------|--------|
| Bondora **LoanData** (~100–112 features), retrieved **2024-01-03** | Features + survival labels; 3-year loans; temporal split 2020-01-01 | **Used:** Kaggle mirror below |
| Bondora **RepaymentsData** | IRR / Table 1 returns | Still unavailable (legacy URL 403) |

### Canonical LoanData source (DOMAIN_02)

- **B01** installs `data/raw/bondora/LoanData.csv` (SHA-256 pinned in `cfg.BONDORA`)
- **Kaggle API download:** https://www.kaggle.com/api/v1/datasets/download/marcobeyer/bondora-p2p-loans
- **Dataset slug:** `marcobeyer/bondora-p2p-loans`
- Fallbacks: existing hash-ok file → Kaggle API (`~/.kaggle/kaggle.json`)
- **Provenance:** `data/raw/bondora/SOURCE.md` + `results/logs/b01_bondora_status.json`
- **Alignment:** `D00_DOMAIN_02_align_loan_vintage.py` → `data/interim/domain2/LoanData_aligned.*`

### Rejected / superseded locals

1. `loan_dataset_investor.xlsx` — current portal, **~31 columns** (wrong schema).
2. Internet Archive `LoanData.zip` **2018-08-25** — schema OK but no 2020 test year (legacy fallback only).
3. Wayback 2020 dumps — truncated / unusable.

## Pipeline order (DOMAIN_02)

`D00` align → `D01` features → `D02` split (test = **2020 only**) → `D03` Cox linear + XGB-Cox/Optuna → `D04` reproduction table.

**Fase A: complete.** Paper assets:
- `results/reproduction/DOMAIN_02_reproduction_table.{json,csv,md,tex}`
- `results/reproduction/DOMAIN_02_paper_asset.md`

IRR remains skipped until RepaymentsData appears (not required for roadmap H1/H3).
