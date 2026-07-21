# Author `results/*.bin` — inspection notes

Downloaded from [habedi/SurvivalAnalysisQACommunities](https://github.com/habedi/SurvivalAnalysisQACommunities) via Git LFS media URLs into `data/raw/stackexchange/author_results/`.

## Contents

Each `runs_info_{θ}_{featureset}.bin` is a pickle:

```text
models_data[(run_id, fold_id)] = (variable_importance: dict[str, float], c_index: float)
```

- 30 runs × 5 folds = **150** entries per cell (matches paper §5.2).
- Featuresets: `b` = behavioural, `cb` = content, `cb+b` = combined.
- **No RSF model object and no hyperparameters** are stored — only permutation importance + fold C-index.

## Bin mean C vs Table 8

Bins are essentially the source of Table 8 (paper rounds to 2 decimals). Gaps `|bin − paper|` ≤ ~0.013 (CS combined θ=36); most ≤ 0.005.

See `data/raw/stackexchange/author_results/author_bin_summary.json`.

## Implication for mimicry

Public notebook defaults (`num_trees=5`, `max_depth=5`, `min_node_size=30`) are the only HP evidence; bins do not reveal a different grid. Closing our gap therefore hinges on **backend (PySurvival concordance)**, **cohort filter**, and **CV quirks**, not on discovering unpublished tree counts from the pickles.
