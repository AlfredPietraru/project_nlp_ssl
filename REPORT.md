# Consolidated Experiment Report

Last updated: 2026-05-26

This document reflects the current codebase and the experiment artifacts stored in this repository. It supersedes older assumptions in `README.md` and `IMPLEMENTATION.md` that describe a DistilBERT/LoRA pipeline.

## Current Pipeline

- Training entrypoint: `train_embedding_classifier.py`
- Core implementation: `ProblemClassificationPipeline.py`
- Dataset access: `DatasetReader.py`
- Model family: sentence-embedding encoder plus shallow multi-task classifier
- Default encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Alternative encoder tested: `sentence-transformers/all-mpnet-base-v2`

The current training flow is:

1. Read `train` and `test` from `dataset_splits.json` via `DatasetReader`.
2. Create a validation split from the train split using a stratified-by-difficulty split with `validation_ratio = 0.15`.
3. Build text inputs from one of:
   - `DESCRIPTION_ONLY`
   - `SOLUTION_ONLY`
   - `DESCRIPTION_AND_SOLUTION`
4. Encode the text with a sentence-transformer model.
5. Train a shared MLP trunk with two heads:
   - difficulty classification
   - multi-label tag prediction
6. Stop early using validation loss.
7. Save metrics, plots, metadata, and a per-experiment markdown report.

## Important Limitation

The current repository saves experiment metadata and plots, but it does not save reloadable model weights anymore. `FrozenEmbeddingPipeline.load()` currently raises a runtime error, so `predict_embedding_classifier.py` is not usable in its present form without restoring checkpoint saving/loading.

## Latest Experiment Snapshot

The most recent experiment artifact in this repo is:

- `rerun_best_balanced_cleaned_mpnet`
- Metadata timestamp: 2026-05-26 21:46
- Encoder: `sentence-transformers/all-mpnet-base-v2`
- Input mode: `DESCRIPTION_AND_SOLUTION`
- Difficulty labels: 8 (`A` through `H`)

Test metrics:

| Metric | Value |
| --- | ---: |
| Difficulty accuracy | 0.3275 |
| Difficulty macro F1 | 0.1925 |
| Tags subset accuracy | 0.0070 |
| Tags micro precision | 0.2240 |
| Tags micro recall | 0.6672 |
| Tags micro F1 | 0.3354 |

Interpretation:

- The latest `mpnet` run improved tag quality over the older cleaned-data baseline.
- It did not beat the best difficulty accuracy result.
- Exact-match tag prediction remains very hard across all runs.

## Best Results Across Recorded Runs

| Goal | Best Experiment | Value |
| --- | --- | ---: |
| Best difficulty accuracy | `rerun_sol_only_cleaned` | 0.3972 |
| Best difficulty macro F1 | `rerun_augmented_mergefgh` | 0.2687 |
| Best tags micro F1 | `rerun_augmented_mergefgh_weighteddiff` | 0.3434 |
| Best tags subset accuracy | `rerun_desc_only_cleaned` and `rerun_best_balanced_cleaned_mpnet` | 0.0070 |
| Best validation loss | `rerun_augmented_mergefgh_unfreeze2` | 1.8333 |

## Summary Table

| Experiment | Dataset | Encoder | Mode | Diff Labels | Best Epoch | Stopped Epoch | Diff Acc | Diff Macro F1 | Tag Subset Acc | Tag Micro F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rerun_best_balanced_cleaned` | `cleaned_data` | `all-MiniLM-L6-v2` | `DESCRIPTION_AND_SOLUTION` | 8 | 8 | 13 | 0.3554 | 0.2128 | 0.0000 | 0.3138 |
| `rerun_best_balanced_cleaned_mpnet` | `cleaned_data` | `all-mpnet-base-v2` | `DESCRIPTION_AND_SOLUTION` | 8 | 8 | 13 | 0.3275 | 0.1925 | 0.0070 | 0.3354 |
| `rerun_desc_only_cleaned` | `cleaned_data` | `all-MiniLM-L6-v2` | `DESCRIPTION_ONLY` | 8 | 7 | 12 | 0.2997 | 0.1799 | 0.0070 | 0.3073 |
| `rerun_sol_only_cleaned` | `cleaned_data` | `all-MiniLM-L6-v2` | `SOLUTION_ONLY` | 8 | 8 | 13 | 0.3972 | 0.2362 | 0.0000 | 0.2823 |
| `rerun_augmented_uncapped` | `augmented_data` | `all-MiniLM-L6-v2` | `DESCRIPTION_AND_SOLUTION` | 8 | 9 | 14 | 0.3467 | 0.2097 | 0.0033 | 0.3214 |
| `rerun_augmented_cap250` | `augmented_data` | `all-MiniLM-L6-v2` | `DESCRIPTION_AND_SOLUTION` | 8 | 13 | 18 | 0.3300 | 0.2299 | 0.0033 | 0.3129 |
| `rerun_augmented_mergefgh` | `augmented_data` | `all-MiniLM-L6-v2` | `DESCRIPTION_AND_SOLUTION` | 6 | 10 | 15 | 0.3100 | 0.2687 | 0.0033 | 0.3287 |
| `rerun_augmented_mergefgh_unfreeze2` | `augmented_data` | `all-MiniLM-L6-v2` | `DESCRIPTION_AND_SOLUTION` | 6 | 5 | 10 | 0.3167 | 0.2684 | 0.0033 | 0.3316 |
| `rerun_augmented_mergefgh_weighteddiff` | `augmented_data` | `all-MiniLM-L6-v2` | `DESCRIPTION_AND_SOLUTION` | 6 | 14 | 19 | 0.2833 | 0.2681 | 0.0033 | 0.3434 |
| `loss_exp_best_balanced_maxlen1024` | `cleaned_data` | `all-MiniLM-L6-v2` | `DESCRIPTION_AND_SOLUTION` | 8 | 8 | 13 | 0.3554 | 0.2128 | 0.0000 | 0.3138 |
| `loss_exp_best_balanced_maxlen2048` | `cleaned_data` | `all-MiniLM-L6-v2` | `DESCRIPTION_AND_SOLUTION` | 8 | 8 | 13 | 0.3554 | 0.2128 | 0.0000 | 0.3138 |

## What Changed Between Runs

### Input ablations on cleaned data

- `SOLUTION_ONLY` gave the best difficulty accuracy: `0.3972`
- `DESCRIPTION_ONLY` gave better exact-match tag accuracy than the MiniLM combined baseline, but only by a very small margin
- `DESCRIPTION_AND_SOLUTION` produced a better balance between tasks than solution-only on cleaned data

### Augmentation and difficulty merging

- Augmented runs increased tag micro F1 relative to the cleaned-data MiniLM baseline
- Merging `G` and `H` into `F` reduced the difficulty task from 8 labels to 6 labels and produced the strongest difficulty macro F1 values
- The weighted-difficulty loss variant traded away difficulty accuracy for the best tag micro F1

### Encoder swap

- `all-mpnet-base-v2` improved tag micro F1 from `0.3138` to `0.3354` on the cleaned balanced setting
- The same run reduced difficulty accuracy from `0.3554` to `0.3275`

## Max Length Findings

The `loss_exp_best_balanced_maxlen1024` and `loss_exp_best_balanced_maxlen2048` experiments did not actually increase usable context length:

| Experiment | Requested Max Len | Effective Max Len |
| --- | ---: | ---: |
| `loss_exp_best_balanced_maxlen1024` | 1024 | 512 |
| `loss_exp_best_balanced_maxlen2048` | 2048 | 512 |

Reason:

- `sentence-transformers/all-MiniLM-L6-v2` is capped at 512 positions in this pipeline, so both runs are metric-identical to the 512-token baseline.

## Practical Takeaways

- If the goal is the best difficulty accuracy, use `rerun_sol_only_cleaned`.
- If the goal is the best difficulty macro F1 under the merged-classes setup, use `rerun_augmented_mergefgh`.
- If the goal is the best tag micro F1, use `rerun_augmented_mergefgh_weighteddiff`.
- If the goal is the best cleaned-data tag performance without class merging, `rerun_best_balanced_cleaned_mpnet` is the strongest current option.

## Per-Experiment Reports

- `rerun_best_balanced_cleaned/experiment_report.md`
- `rerun_best_balanced_cleaned_mpnet/experiment_report.md`
- `rerun_desc_only_cleaned/experiment_report.md`
- `rerun_sol_only_cleaned/experiment_report.md`
- `rerun_augmented_uncapped/experiment_report.md`
- `rerun_augmented_cap250/experiment_report.md`
- `rerun_augmented_mergefgh/experiment_report.md`
- `rerun_augmented_mergefgh_unfreeze2/experiment_report.md`
- `rerun_augmented_mergefgh_weighteddiff/experiment_report.md`
- `loss_exp_best_balanced_maxlen1024/experiment_report.md`
- `loss_exp_best_balanced_maxlen2048/experiment_report.md`
