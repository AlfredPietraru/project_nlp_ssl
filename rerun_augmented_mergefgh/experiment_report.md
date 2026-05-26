# Experiment Report

## Dataset Construction
- Dataset root: `augmented_data`
- Running mode: `DESCRIPTION_AND_SOLUTION`
- Merge `G/H -> F`: `True`
- Train cap per difficulty: `None`
- Train/validation split: stratified by difficulty from the train split
- Validation ratio per difficulty bucket: `0.15`
- Test split: original dataset test split, untouched
- Random solution sampling in train: `True`
- Deterministic first solution in validation/test: `True`

## Model
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Requested max length: `512`
- Effective max length used: `512`
- Model-supported max length: `512`
- Unfrozen top transformer layers: `0`
- Encoder learning rate: `1e-05`
- Head hidden dim: `256`
- Dropout: `0.2`

## Training Hyperparameters
- Batch size: `32`
- Head learning rate: `0.001`
- Weight decay: `0.0001`
- Difficulty loss type: `cross_entropy`
- Tags loss type: `weighted_bce`
- Difficulty loss weight: `1.0`
- Tags loss weight: `0.5`
- Early stopping patience: `5`
- Early stopping min delta: `0.01`
- Max epochs: `40`
- Best epoch: `10`
- Stopped epoch: `15`
- Best validation loss: `1.8924`

## Difficulty Metrics
| Metric | Value |
| --- | --- |
| Accuracy | 0.3100 |
| Macro F1 | 0.2687 |

## Tag Metrics
| Metric | Value |
| --- | --- |
| Subset Accuracy | 0.0033 |
| Micro Precision | 0.2240 |
| Micro Recall | 0.6178 |
| Micro F1 | 0.3287 |

## Difficulty Per-Class Accuracy
| Difficulty | Support | Accuracy |
| --- | --- | --- |
| A | 71 | 0.6338 |
| B | 64 | 0.2344 |
| C | 49 | 0.0612 |
| D | 48 | 0.1042 |
| E | 42 | 0.4286 |
| F | 26 | 0.2692 |

## Label Statistics
- Number of training examples after validation split: `1438`
- Difficulty counts: `[341, 311, 235, 229, 203, 119]`
- Number of tags: `35`

## Generated Files
- `metadata.json`
- `training_validation_loss.png`
- `training_validation_difficulty_accuracy.png`
- `training_validation_tags_micro_f1.png`
