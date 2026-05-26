#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ProblemClassificationPipeline import FrozenEmbeddingPipeline, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a frozen-embedding classifier for problem difficulty and tags.",
    )
    parser.add_argument("--dataset-root", default="cleaned_data")
    parser.add_argument("--output-dir", default="embedding_classifier_output")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--running-mode", default="DESCRIPTION_AND_SOLUTION")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--tag-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--difficulty-loss-weight", type=float, default=1.0)
    parser.add_argument("--tags-loss-weight", type=float, default=0.5)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--difficulty-loss-type", default="cross_entropy", choices=["cross_entropy", "weighted_cross_entropy"])
    parser.add_argument("--tags-loss-type", default="weighted_bce", choices=["bce", "weighted_bce"])
    parser.add_argument("--max-difficulty-class-weight", type=float, default=10.0)
    parser.add_argument("--max-tag-pos-weight", type=float, default=20.0)
    parser.add_argument("--max-train-examples-per-difficulty", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.01)
    parser.add_argument("--merge-gh-into-f", action="store_true")
    parser.add_argument("--unfreeze-top-n-transformer-layers", type=int, default=0)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        embedding_model_name=args.embedding_model,
        running_mode=args.running_mode,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_epochs=args.num_epochs,
        validation_ratio=args.validation_ratio,
        random_seed=args.random_seed,
        tag_threshold=args.tag_threshold,
        num_workers=args.num_workers,
        difficulty_loss_weight=args.difficulty_loss_weight,
        tags_loss_weight=args.tags_loss_weight,
        embedding_max_length=args.embedding_max_length,
        device=args.device,
        difficulty_loss_type=args.difficulty_loss_type,
        tags_loss_type=args.tags_loss_type,
        max_difficulty_class_weight=args.max_difficulty_class_weight,
        max_tag_pos_weight=args.max_tag_pos_weight,
        max_train_examples_per_difficulty=args.max_train_examples_per_difficulty,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        merge_gh_into_f=args.merge_gh_into_f,
        unfreeze_top_n_transformer_layers=args.unfreeze_top_n_transformer_layers,
        encoder_learning_rate=args.encoder_learning_rate,
    )

    pipeline = FrozenEmbeddingPipeline(config)
    metadata = pipeline.train()
    print(json.dumps(metadata["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
