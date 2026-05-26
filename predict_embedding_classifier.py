#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ProblemClassificationPipeline import FrozenEmbeddingPipeline, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained frozen-embedding classifier.",
    )
    parser.add_argument("--dataset-root", default="cleaned_data")
    parser.add_argument("--model-dir", default="embedding_classifier_output")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--running-mode", default="DESCRIPTION_AND_SOLUTION")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tag-threshold", type=float, default=0.5)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=args.model_dir,
        embedding_model_name=args.embedding_model,
        running_mode=args.running_mode,
        batch_size=args.batch_size,
        tag_threshold=args.tag_threshold,
        embedding_max_length=args.embedding_max_length,
        device=args.device,
    )

    pipeline = FrozenEmbeddingPipeline(config)
    predictions = pipeline.predict_split(split_name=args.split, model_dir=args.model_dir)
    print(json.dumps({"num_predictions": len(predictions)}, indent=2))


if __name__ == "__main__":
    main()
