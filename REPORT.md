# Consolidated Experiment Report

STEFAN TAKE A LOOK ON THIS!!!!!

## Best Runs
- Best difficulty accuracy: `rerun_sol_only_cleaned` with `0.3972`
- Best difficulty macro F1: `rerun_augmented_mergefgh` with `0.2687`
- Best tags micro F1: `rerun_augmented_mergefgh_weighteddiff` with `0.3434`
- Best tags subset accuracy: `rerun_desc_only_cleaned` with `0.0070`

## Summary Table
| Experiment | Requested Max Len | Effective Max Len | Best Epoch | Stopped Epoch | Difficulty Acc | Difficulty Macro F1 | Tags Subset Acc | Tags Micro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rerun_best_balanced_cleaned | 512 | 512 | 8 | 13 | 0.3554 | 0.2128 | 0.0000 | 0.3138 |
| rerun_desc_only_cleaned | 512 | 512 | 7 | 12 | 0.2997 | 0.1799 | 0.0070 | 0.3073 |
| rerun_sol_only_cleaned | 512 | 512 | 8 | 13 | 0.3972 | 0.2362 | 0.0000 | 0.2823 |
| rerun_augmented_uncapped | 512 | 512 | 9 | 14 | 0.3467 | 0.2097 | 0.0033 | 0.3214 |
| rerun_augmented_cap250 | 512 | 512 | 13 | 18 | 0.3300 | 0.2299 | 0.0033 | 0.3129 |
| rerun_augmented_mergefgh | 512 | 512 | 10 | 15 | 0.3100 | 0.2687 | 0.0033 | 0.3287 |
| rerun_augmented_mergefgh_weighteddiff | 512 | 512 | 14 | 19 | 0.2833 | 0.2681 | 0.0033 | 0.3434 |
| rerun_augmented_mergefgh_unfreeze2 | 512 | 512 | 5 | 10 | 0.3167 | 0.2684 | 0.0033 | 0.3316 |
| loss_exp_best_balanced_maxlen1024 | 1024 | 512 | 8 | 13 | 0.3554 | 0.2128 | 0.0000 | 0.3138 |
| loss_exp_best_balanced_maxlen2048 | 2048 | 512 | 8 | 13 | 0.3554 | 0.2128 | 0.0000 | 0.3138 |

## Tag Metrics and Reports
| Experiment | Tags Precision | Tags Recall | Tags Micro F1 | Detailed Report |
| --- | --- | --- | --- | --- |
| rerun_best_balanced_cleaned | 0.2107 | 0.6146 | 0.3138 | rerun_best_balanced_cleaned/experiment_report.md |
| rerun_desc_only_cleaned | 0.2080 | 0.5882 | 0.3073 | rerun_desc_only_cleaned/experiment_report.md |
| rerun_sol_only_cleaned | 0.1797 | 0.6579 | 0.2823 | rerun_sol_only_cleaned/experiment_report.md |
| rerun_augmented_uncapped | 0.2161 | 0.6267 | 0.3214 | rerun_augmented_uncapped/experiment_report.md |
| rerun_augmented_cap250 | 0.2079 | 0.6326 | 0.3129 | rerun_augmented_cap250/experiment_report.md |
| rerun_augmented_mergefgh | 0.2240 | 0.6178 | 0.3287 | rerun_augmented_mergefgh/experiment_report.md |
| rerun_augmented_mergefgh_weighteddiff | 0.2312 | 0.6667 | 0.3434 | rerun_augmented_mergefgh_weighteddiff/experiment_report.md |
| rerun_augmented_mergefgh_unfreeze2 | 0.2287 | 0.6030 | 0.3316 | rerun_augmented_mergefgh_unfreeze2/experiment_report.md |
| loss_exp_best_balanced_maxlen1024 | 0.2107 | 0.6146 | 0.3138 | loss_exp_best_balanced_maxlen1024/experiment_report.md |
| loss_exp_best_balanced_maxlen2048 | 0.2107 | 0.6146 | 0.3138 | loss_exp_best_balanced_maxlen2048/experiment_report.md |

## Notes
- The `loss_exp_best_balanced_maxlen1024` and `loss_exp_best_balanced_maxlen2048` requests still ran with effective max length `512` because `all-MiniLM-L6-v2` only supports `512` positions.
- Detailed per-class difficulty metrics, hyperparameters, split logic, and train/validation plots live inside each experiment folder's `experiment_report.md`.

