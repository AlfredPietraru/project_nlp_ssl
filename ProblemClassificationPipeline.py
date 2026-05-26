from __future__ import annotations

import json
import math
import random
import copy
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from DatasetReader import DatasetEntry, DatasetReader
from ProgrammingProblemsDataset import (
    ProgrammingProblemsDataLoader,
    ProgrammingProblemsDataset,
    RunningMode,
    TextEmbeddingModel,
)


DIFFICULTY_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]


@dataclass
class TrainingConfig:
    dataset_root: str = "cleaned_data"
    output_dir: str = "embedding_classifier_output"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    running_mode: str = RunningMode.DESCRIPTION_AND_SOLUTION.value
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    dropout: float = 0.2
    num_epochs: int = 15
    validation_ratio: float = 0.15
    random_seed: int = 42
    tag_threshold: float = 0.5
    num_workers: int = 0
    difficulty_loss_weight: float = 1.0
    tags_loss_weight: float = 0.5
    embedding_max_length: int = 512
    device: str | None = None
    difficulty_loss_type: str = "cross_entropy"
    tags_loss_type: str = "weighted_bce"
    max_difficulty_class_weight: float = 10.0
    max_tag_pos_weight: float = 20.0
    max_train_examples_per_difficulty: int | None = None
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.01
    merge_gh_into_f: bool = False
    unfreeze_top_n_transformer_layers: int = 0
    encoder_learning_rate: float = 1e-5


class ProblemLabelEncoder:
    def __init__(self, difficulties: list[str], tags: list[str]) -> None:
        self.difficulties = difficulties
        self.tags = tags
        self._difficulty_to_index = {label: index for index, label in enumerate(difficulties)}
        self._tag_to_index = {label: index for index, label in enumerate(tags)}

    @classmethod
    def fit(cls, entries: list[DatasetEntry]) -> "ProblemLabelEncoder":
        difficulty_set = {entry.difficulty.strip().upper() for entry in entries if entry.difficulty}
        ordered_difficulties = [label for label in DIFFICULTY_ORDER if label in difficulty_set]
        remaining = sorted(difficulty_set - set(ordered_difficulties))
        difficulties = ordered_difficulties + remaining

        tag_set = set()
        for entry in entries:
            tag_set.update(tag for tag in entry.tags if tag)

        return cls(difficulties=difficulties, tags=sorted(tag_set))

    def encode_difficulty(self, difficulty: str) -> int:
        normalized = difficulty.strip().upper()
        if normalized not in self._difficulty_to_index:
            raise KeyError(f"Unknown difficulty label: {difficulty}")
        return self._difficulty_to_index[normalized]

    def decode_difficulty(self, index: int) -> str:
        return self.difficulties[index]

    def encode_tags(self, tags: list[str]) -> torch.Tensor:
        vector = torch.zeros(len(self.tags), dtype=torch.float32)
        for tag in tags:
            if tag in self._tag_to_index:
                vector[self._tag_to_index[tag]] = 1.0
        return vector

    def decode_tags(self, scores: torch.Tensor, threshold: float) -> list[str]:
        return [
            tag
            for tag, score in zip(self.tags, scores.tolist())
            if score >= threshold
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "difficulties": self.difficulties,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProblemLabelEncoder":
        return cls(
            difficulties=list(data.get("difficulties", [])),
            tags=list(data.get("tags", [])),
        )


@dataclass
class LabelStatistics:
    difficulty_counts: list[int]
    tag_positive_counts: list[int]
    num_examples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EncodedProblemsDataset(Dataset):
    def __init__(
        self,
        base_dataset: ProgrammingProblemsDataset,
        label_encoder: ProblemLabelEncoder,
    ) -> None:
        self.base_dataset = base_dataset
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base_dataset[index]
        return {
            "id": item["id"],
            "text": item["text"],
            "embedding": item["embedding"].float(),
            "difficulty_label": torch.tensor(
                self.label_encoder.encode_difficulty(item["difficulty"]),
                dtype=torch.long,
            ),
            "tag_labels": self.label_encoder.encode_tags(item["tags"]),
            "difficulty": item["difficulty"],
            "tags": item["tags"],
        }


class TextProblemsDataset(Dataset):
    def __init__(
        self,
        base_dataset: ProgrammingProblemsDataset,
        label_encoder: ProblemLabelEncoder,
    ) -> None:
        self.base_dataset = base_dataset
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base_dataset[index]
        return {
            "id": item["id"],
            "text": item["text"],
            "difficulty_label": torch.tensor(
                self.label_encoder.encode_difficulty(item["difficulty"]),
                dtype=torch.long,
            ),
            "tag_labels": self.label_encoder.encode_tags(item["tags"]),
            "difficulty": item["difficulty"],
            "tags": item["tags"],
        }


class EncodedProblemsDataLoader(ProgrammingProblemsDataLoader):
    @staticmethod
    def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": [item["id"] for item in batch],
            "text": [item["text"] for item in batch],
            "difficulty": [item["difficulty"] for item in batch],
            "tags": [item["tags"] for item in batch],
            "embedding": torch.stack([item["embedding"] for item in batch]),
            "difficulty_label": torch.stack([item["difficulty_label"] for item in batch]),
            "tag_labels": torch.stack([item["tag_labels"] for item in batch]),
        }


class TextProblemsDataLoader(ProgrammingProblemsDataLoader):
    @staticmethod
    def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": [item["id"] for item in batch],
            "text": [item["text"] for item in batch],
            "difficulty": [item["difficulty"] for item in batch],
            "tags": [item["tags"] for item in batch],
            "difficulty_label": torch.stack([item["difficulty_label"] for item in batch]),
            "tag_labels": torch.stack([item["tag_labels"] for item in batch]),
        }


class FrozenEmbeddingClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_difficulties: int,
        num_tags: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.difficulty_head = nn.Linear(hidden_dim, num_difficulties)
        self.tags_head = nn.Linear(hidden_dim, num_tags)

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.shared(embedding)
        return {
            "difficulty_logits": self.difficulty_head(features),
            "tags_logits": self.tags_head(features),
        }


class TransformerFineTuningClassifier(nn.Module):
    def __init__(
        self,
        embedding_model: TextEmbeddingModel,
        num_difficulties: int,
        num_tags: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self.shared = nn.Sequential(
            nn.Linear(embedding_model.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.difficulty_head = nn.Linear(hidden_dim, num_difficulties)
        self.tags_head = nn.Linear(hidden_dim, num_tags)

    def forward(self, texts: list[str]) -> dict[str, torch.Tensor]:
        inputs = self.embedding_model.tokenize_batch(texts)
        outputs = self.embedding_model.model(**inputs)
        embedding = self.embedding_model.mean_pool(
            token_embeddings=outputs.last_hidden_state,
            attention_mask=inputs["attention_mask"],
        )
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        features = self.shared(embedding)
        return {
            "difficulty_logits": self.difficulty_head(features),
            "tags_logits": self.tags_head(features),
        }


class ProblemDatasetFactory:
    def __init__(
        self,
        dataset_reader: DatasetReader,
        embedding_model: TextEmbeddingModel,
        running_mode: RunningMode,
        validation_ratio: float,
        random_seed: int,
    ) -> None:
        self.dataset_reader = dataset_reader
        self.embedding_model = embedding_model
        self.running_mode = running_mode
        self.validation_ratio = validation_ratio
        self.random_seed = random_seed
        self.max_train_examples_per_difficulty: int | None = None
        self.merge_gh_into_f = False

    def set_max_train_examples_per_difficulty(self, value: int | None) -> None:
        self.max_train_examples_per_difficulty = value

    def set_merge_gh_into_f(self, value: bool) -> None:
        self.merge_gh_into_f = value

    def build(self) -> tuple[
        ProgrammingProblemsDataset,
        ProgrammingProblemsDataset,
        ProgrammingProblemsDataset,
        ProblemLabelEncoder,
        LabelStatistics,
    ]:
        train_entries = self._normalize_entries(
            self.dataset_reader.retrieve_split_data("train"),
        )
        test_entries = self._normalize_entries(
            self.dataset_reader.retrieve_split_data("test"),
        )

        if self.max_train_examples_per_difficulty is not None:
            train_entries = self._cap_train_entries(
                train_entries,
                max_examples_per_difficulty=self.max_train_examples_per_difficulty,
            )

        split_train_entries, split_validation_entries = self._split_train_validation(train_entries)

        label_encoder = ProblemLabelEncoder.fit(split_train_entries)
        label_statistics = self._compute_label_statistics(split_train_entries, label_encoder)

        train_dataset = self._build_dataset(split_train_entries, sample_random_solution=True)
        validation_dataset = self._build_dataset(split_validation_entries, sample_random_solution=False)
        test_dataset = self._build_dataset(test_entries, sample_random_solution=False)

        return train_dataset, validation_dataset, test_dataset, label_encoder, label_statistics

    def _build_dataset(
        self,
        entries: list[DatasetEntry],
        sample_random_solution: bool,
    ) -> ProgrammingProblemsDataset:
        return ProgrammingProblemsDataset(
            dataset_reader=self.dataset_reader,
            is_train=sample_random_solution,
            running_mode=self.running_mode,
            embedding_model=self.embedding_model,
            include_embeddings=False,
            entries=entries,
            sample_random_solution=sample_random_solution,
        )

    def _split_train_validation(self, entries: list[DatasetEntry]) -> tuple[list[DatasetEntry], list[DatasetEntry]]:
        grouped_entries: dict[str, list[DatasetEntry]] = defaultdict(list)
        for entry in entries:
            grouped_entries[entry.difficulty.strip().upper()].append(entry)

        train_entries: list[DatasetEntry] = []
        validation_entries: list[DatasetEntry] = []
        random_generator = random.Random(self.random_seed)

        for difficulty, difficulty_entries in grouped_entries.items():
            shuffled = list(difficulty_entries)
            random_generator.shuffle(shuffled)

            validation_count = int(math.ceil(len(shuffled) * self.validation_ratio))
            if len(shuffled) > 1:
                validation_count = max(1, min(validation_count, len(shuffled) - 1))
            else:
                validation_count = 0

            validation_entries.extend(shuffled[:validation_count])
            train_entries.extend(shuffled[validation_count:])

        random_generator.shuffle(train_entries)
        random_generator.shuffle(validation_entries)
        return train_entries, validation_entries

    @staticmethod
    def _compute_label_statistics(
        entries: list[DatasetEntry],
        label_encoder: ProblemLabelEncoder,
    ) -> LabelStatistics:
        difficulty_counts = [0 for _ in label_encoder.difficulties]
        tag_positive_counts = [0 for _ in label_encoder.tags]

        for entry in entries:
            difficulty_counts[label_encoder.encode_difficulty(entry.difficulty)] += 1
            for tag in entry.tags:
                tag_index = label_encoder._tag_to_index.get(tag)
                if tag_index is not None:
                    tag_positive_counts[tag_index] += 1

        return LabelStatistics(
            difficulty_counts=difficulty_counts,
            tag_positive_counts=tag_positive_counts,
            num_examples=len(entries),
        )

    def _cap_train_entries(
        self,
        entries: list[DatasetEntry],
        max_examples_per_difficulty: int,
    ) -> list[DatasetEntry]:
        grouped_entries: dict[str, list[DatasetEntry]] = defaultdict(list)
        for entry in entries:
            grouped_entries[entry.difficulty.strip().upper()].append(entry)

        random_generator = random.Random(self.random_seed)
        capped_entries: list[DatasetEntry] = []

        for difficulty_entries in grouped_entries.values():
            shuffled = list(difficulty_entries)
            random_generator.shuffle(shuffled)
            capped_entries.extend(shuffled[:max_examples_per_difficulty])

        random_generator.shuffle(capped_entries)
        return capped_entries

    def _normalize_entries(self, entries: list[DatasetEntry]) -> list[DatasetEntry]:
        if not self.merge_gh_into_f:
            return entries

        normalized_entries: list[DatasetEntry] = []
        for entry in entries:
            normalized_entry = copy.copy(entry)
            difficulty = entry.difficulty.strip().upper()
            if difficulty in {"G", "H"}:
                normalized_entry.difficulty = "F"
            normalized_entries.append(normalized_entry)
        return normalized_entries


class FrozenEmbeddingPipeline:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_reader = DatasetReader(Path(config.dataset_root))
        self.running_mode = RunningMode(config.running_mode)
        self.embedding_model = TextEmbeddingModel(
            model_name=config.embedding_model_name,
            max_length=config.embedding_max_length,
            device=config.device,
        )

    def train(self) -> dict[str, Any]:
        self._set_seed(self.config.random_seed)

        dataset_factory = ProblemDatasetFactory(
            dataset_reader=self.dataset_reader,
            embedding_model=self.embedding_model,
            running_mode=self.running_mode,
            validation_ratio=self.config.validation_ratio,
            random_seed=self.config.random_seed,
        )
        dataset_factory.set_max_train_examples_per_difficulty(
            self.config.max_train_examples_per_difficulty,
        )
        dataset_factory.set_merge_gh_into_f(self.config.merge_gh_into_f)
        train_base_dataset, validation_base_dataset, test_base_dataset, label_encoder, label_statistics = dataset_factory.build()

        model, train_loader, validation_loader, test_loader, input_dim, optimizer = self._build_training_components(
            train_base_dataset=train_base_dataset,
            validation_base_dataset=validation_base_dataset,
            test_base_dataset=test_base_dataset,
            label_encoder=label_encoder,
        )

        difficulty_class_weights = self._build_difficulty_class_weights(label_statistics)
        tag_pos_weights = self._build_tag_pos_weights(label_statistics)

        best_validation_loss = float("inf")
        best_model_path = self.output_dir / "best_model.pt"
        history: list[dict[str, Any]] = []
        best_epoch = 0
        early_stopping_counter = 0

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self._run_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                is_training=True,
                difficulty_class_weights=difficulty_class_weights,
                tag_pos_weights=tag_pos_weights,
            )
            validation_metrics = self._run_epoch(
                model=model,
                data_loader=validation_loader,
                optimizer=None,
                is_training=False,
                difficulty_class_weights=difficulty_class_weights,
                tag_pos_weights=tag_pos_weights,
            )

            epoch_metrics = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(epoch_metrics)

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={validation_metrics['loss']:.4f} "
                f"val_diff_acc={validation_metrics['difficulty_accuracy']:.4f} "
                f"val_tag_f1={validation_metrics['tags_micro_f1']:.4f}"
            )

            if validation_metrics["loss"] <= best_validation_loss - self.config.early_stopping_min_delta:
                best_validation_loss = validation_metrics["loss"]
                best_epoch = epoch
                early_stopping_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= self.config.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch:02d} | "
                    f"best_epoch={best_epoch:02d} "
                    f"best_val_loss={best_validation_loss:.4f}"
                )
                break

        model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        test_metrics = self._run_epoch(
            model=model,
            data_loader=test_loader,
            optimizer=None,
            is_training=False,
            difficulty_class_weights=difficulty_class_weights,
            tag_pos_weights=tag_pos_weights,
        )

        metadata = {
            "config": asdict(self.config),
            "running_mode": self.running_mode.value,
            "merge_gh_into_f": self.config.merge_gh_into_f,
            "unfreeze_top_n_transformer_layers": self.config.unfreeze_top_n_transformer_layers,
            "input_dim": input_dim,
            "label_encoder": label_encoder.to_dict(),
            "label_statistics": label_statistics.to_dict(),
            "difficulty_class_weights": difficulty_class_weights.detach().cpu().tolist(),
            "tag_pos_weights": tag_pos_weights.detach().cpu().tolist(),
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "stopped_epoch": len(history),
            "history": history,
            "test_metrics": test_metrics,
        }
        self._save_artifacts(model=model, metadata=metadata)
        return metadata

    def predict_split(self, split_name: str = "test", model_dir: str | Path | None = None) -> dict[str, Any]:
        model, metadata, label_encoder = self.load(model_dir=model_dir)
        data_loader = self._build_prediction_loader(split_name=split_name, label_encoder=label_encoder)

        predictions: dict[str, Any] = {}
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                embeddings = batch["embedding"].to(self.device)
                outputs = model(embeddings)
                difficulty_probs = torch.softmax(outputs["difficulty_logits"], dim=1)
                tag_probs = torch.sigmoid(outputs["tags_logits"])

                for index, problem_id in enumerate(batch["id"]):
                    difficulty_index = int(torch.argmax(difficulty_probs[index]).item())
                    difficulty_scores = difficulty_probs[index].detach().cpu()
                    tag_scores = tag_probs[index].detach().cpu()
                    predictions[problem_id] = {
                        "difficulty": label_encoder.decode_difficulty(difficulty_index),
                        "difficulty_confidence": float(difficulty_scores[difficulty_index].item()),
                        "tags": label_encoder.decode_tags(tag_scores, threshold=self.config.tag_threshold),
                        "all_difficulties": {
                            label_encoder.decode_difficulty(label_index): float(score.item())
                            for label_index, score in enumerate(difficulty_scores)
                        },
                        "all_tag_scores": {
                            tag: float(score)
                            for tag, score in zip(label_encoder.tags, tag_scores.tolist())
                        },
                    }

        output_path = Path(model_dir or self.output_dir) / f"{split_name}_predictions.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(predictions, output_file, indent=2)
        return predictions

    def load(
        self,
        model_dir: str | Path | None = None,
    ) -> tuple[FrozenEmbeddingClassifier, dict[str, Any], ProblemLabelEncoder]:
        model_path = Path(model_dir or self.output_dir)
        metadata_path = model_path / "metadata.json"
        weights_path = model_path / "best_model.pt"

        with metadata_path.open(encoding="utf-8") as input_file:
            metadata = json.load(input_file)

        label_encoder = ProblemLabelEncoder.from_dict(metadata["label_encoder"])
        if metadata.get("unfreeze_top_n_transformer_layers", 0) > 0:
            self.embedding_model.freeze_all_parameters()
            self.embedding_model.unfreeze_top_transformer_layers(metadata["unfreeze_top_n_transformer_layers"])
            model = TransformerFineTuningClassifier(
                embedding_model=self.embedding_model,
                num_difficulties=len(label_encoder.difficulties),
                num_tags=len(label_encoder.tags),
                hidden_dim=metadata["config"]["hidden_dim"],
                dropout=metadata["config"]["dropout"],
            ).to(self.device)
        else:
            model = FrozenEmbeddingClassifier(
                input_dim=metadata["input_dim"],
                num_difficulties=len(label_encoder.difficulties),
                num_tags=len(label_encoder.tags),
                hidden_dim=metadata["config"]["hidden_dim"],
                dropout=metadata["config"]["dropout"],
            ).to(self.device)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        return model, metadata, label_encoder

    def _build_prediction_loader(
        self,
        split_name: str,
        label_encoder: ProblemLabelEncoder,
    ) -> EncodedProblemsDataLoader:
        entries = self.dataset_reader.retrieve_split_data(split_name)
        if self.config.merge_gh_into_f:
            entries = ProblemDatasetFactory(
                dataset_reader=self.dataset_reader,
                embedding_model=self.embedding_model,
                running_mode=self.running_mode,
                validation_ratio=self.config.validation_ratio,
                random_seed=self.config.random_seed,
            )._normalize_entries(entries)
        dataset = ProgrammingProblemsDataset(
            dataset_reader=self.dataset_reader,
            is_train=False,
            running_mode=self.running_mode,
            embedding_model=self.embedding_model,
            include_embeddings=self.config.unfreeze_top_n_transformer_layers == 0,
            entries=entries,
            sample_random_solution=False,
        )
        if self.config.unfreeze_top_n_transformer_layers > 0:
            text_dataset = TextProblemsDataset(base_dataset=dataset, label_encoder=label_encoder)
            return TextProblemsDataLoader(
                dataset=text_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
        encoded_dataset = EncodedProblemsDataset(base_dataset=dataset, label_encoder=label_encoder)
        return EncodedProblemsDataLoader(
            dataset=encoded_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def _run_epoch(
        self,
        model: FrozenEmbeddingClassifier,
        data_loader: EncodedProblemsDataLoader,
        optimizer: torch.optim.Optimizer | None,
        is_training: bool,
        difficulty_class_weights: torch.Tensor,
        tag_pos_weights: torch.Tensor,
    ) -> dict[str, float]:
        if is_training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_examples = 0
        total_correct_difficulty = 0
        total_tag_tp = 0.0
        total_tag_fp = 0.0
        total_tag_fn = 0.0
        total_tag_exact_match = 0
        per_class_total = [0 for _ in range(model.difficulty_head.out_features)]
        per_class_correct = [0 for _ in range(model.difficulty_head.out_features)]
        per_class_predicted = [0 for _ in range(model.difficulty_head.out_features)]

        for batch in data_loader:
            difficulty_labels = batch["difficulty_label"].to(self.device)
            tag_labels = batch["tag_labels"].to(self.device)

            if optimizer is not None:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_training):
                outputs = self._forward_model(model, batch)
                difficulty_loss = self._compute_difficulty_loss(
                    logits=outputs["difficulty_logits"],
                    labels=difficulty_labels,
                    class_weights=difficulty_class_weights,
                )
                tags_loss = self._compute_tags_loss(
                    logits=outputs["tags_logits"],
                    labels=tag_labels,
                    pos_weights=tag_pos_weights,
                )
                loss = (
                    self.config.difficulty_loss_weight * difficulty_loss
                    + self.config.tags_loss_weight * tags_loss
                )

                if optimizer is not None:
                    loss.backward()
                    optimizer.step()

            batch_size = difficulty_labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            difficulty_predictions = torch.argmax(outputs["difficulty_logits"], dim=1)
            total_correct_difficulty += int((difficulty_predictions == difficulty_labels).sum().item())
            for label_value, pred_value in zip(difficulty_labels.tolist(), difficulty_predictions.tolist()):
                per_class_total[label_value] += 1
                per_class_predicted[pred_value] += 1
                if label_value == pred_value:
                    per_class_correct[label_value] += 1

            tag_predictions = (torch.sigmoid(outputs["tags_logits"]) >= self.config.tag_threshold).float()
            total_tag_tp += float((tag_predictions * tag_labels).sum().item())
            total_tag_fp += float((tag_predictions * (1.0 - tag_labels)).sum().item())
            total_tag_fn += float(((1.0 - tag_predictions) * tag_labels).sum().item())
            total_tag_exact_match += int((tag_predictions == tag_labels).all(dim=1).sum().item())

        precision = total_tag_tp / (total_tag_tp + total_tag_fp) if (total_tag_tp + total_tag_fp) else 0.0
        recall = total_tag_tp / (total_tag_tp + total_tag_fn) if (total_tag_tp + total_tag_fn) else 0.0
        micro_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        difficulty_macro_f1 = self._compute_macro_f1(
            true_counts=per_class_total,
            predicted_counts=per_class_predicted,
            true_positive_counts=per_class_correct,
        )
        per_class_accuracy = {
            DIFFICULTY_ORDER[index] if index < len(DIFFICULTY_ORDER) else str(index): (
                per_class_correct[index] / per_class_total[index] if per_class_total[index] else 0.0
            )
            for index in range(len(per_class_total))
        }
        per_class_support = {
            DIFFICULTY_ORDER[index] if index < len(DIFFICULTY_ORDER) else str(index): per_class_total[index]
            for index in range(len(per_class_total))
        }

        return {
            "loss": total_loss / total_examples if total_examples else 0.0,
            "difficulty_accuracy": total_correct_difficulty / total_examples if total_examples else 0.0,
            "difficulty_macro_f1": difficulty_macro_f1,
            "tags_micro_precision": precision,
            "tags_micro_recall": recall,
            "tags_micro_f1": micro_f1,
            "tags_subset_accuracy": total_tag_exact_match / total_examples if total_examples else 0.0,
            "difficulty_per_class_accuracy": per_class_accuracy,
            "difficulty_per_class_support": per_class_support,
        }

    def _build_training_components(
        self,
        train_base_dataset: ProgrammingProblemsDataset,
        validation_base_dataset: ProgrammingProblemsDataset,
        test_base_dataset: ProgrammingProblemsDataset,
        label_encoder: ProblemLabelEncoder,
    ) -> tuple[
        nn.Module,
        ProgrammingProblemsDataLoader,
        ProgrammingProblemsDataLoader,
        ProgrammingProblemsDataLoader,
        int,
        torch.optim.Optimizer,
    ]:
        if self.config.unfreeze_top_n_transformer_layers > 0:
            self.embedding_model.freeze_all_parameters()
            self.embedding_model.unfreeze_top_transformer_layers(
                self.config.unfreeze_top_n_transformer_layers,
            )
            model = TransformerFineTuningClassifier(
                embedding_model=self.embedding_model,
                num_difficulties=len(label_encoder.difficulties),
                num_tags=len(label_encoder.tags),
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
            ).to(self.device)
            train_loader = TextProblemsDataLoader(
                dataset=TextProblemsDataset(train_base_dataset, label_encoder),
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )
            validation_loader = TextProblemsDataLoader(
                dataset=TextProblemsDataset(validation_base_dataset, label_encoder),
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
            test_loader = TextProblemsDataLoader(
                dataset=TextProblemsDataset(test_base_dataset, label_encoder),
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
            encoder_params = [param for param in self.embedding_model.model.parameters() if param.requires_grad]
            head_params = [
                param
                for name, param in model.named_parameters()
                if not name.startswith("embedding_model.model.") and param.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": encoder_params, "lr": self.config.encoder_learning_rate},
                    {"params": head_params, "lr": self.config.learning_rate},
                ],
                weight_decay=self.config.weight_decay,
            )
            return model, train_loader, validation_loader, test_loader, self.embedding_model.hidden_size, optimizer

        embedding_dataset_train = ProgrammingProblemsDataset(
            dataset_reader=train_base_dataset.dataset_reader,
            is_train=train_base_dataset.is_train,
            running_mode=train_base_dataset.running_mode,
            embedding_model=self.embedding_model,
            include_embeddings=True,
            entries=train_base_dataset.entries,
            sample_random_solution=train_base_dataset.sample_random_solution,
        )
        embedding_dataset_validation = ProgrammingProblemsDataset(
            dataset_reader=validation_base_dataset.dataset_reader,
            is_train=validation_base_dataset.is_train,
            running_mode=validation_base_dataset.running_mode,
            embedding_model=self.embedding_model,
            include_embeddings=True,
            entries=validation_base_dataset.entries,
            sample_random_solution=validation_base_dataset.sample_random_solution,
        )
        embedding_dataset_test = ProgrammingProblemsDataset(
            dataset_reader=test_base_dataset.dataset_reader,
            is_train=test_base_dataset.is_train,
            running_mode=test_base_dataset.running_mode,
            embedding_model=self.embedding_model,
            include_embeddings=True,
            entries=test_base_dataset.entries,
            sample_random_solution=test_base_dataset.sample_random_solution,
        )
        train_dataset = EncodedProblemsDataset(embedding_dataset_train, label_encoder)
        validation_dataset = EncodedProblemsDataset(embedding_dataset_validation, label_encoder)
        test_dataset = EncodedProblemsDataset(embedding_dataset_test, label_encoder)

        train_loader = EncodedProblemsDataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        validation_loader = EncodedProblemsDataLoader(
            dataset=validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        test_loader = EncodedProblemsDataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        input_dim = train_dataset[0]["embedding"].numel()
        model = FrozenEmbeddingClassifier(
            input_dim=input_dim,
            num_difficulties=len(label_encoder.difficulties),
            num_tags=len(label_encoder.tags),
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return model, train_loader, validation_loader, test_loader, input_dim, optimizer

    def _forward_model(self, model: nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.config.unfreeze_top_n_transformer_layers > 0:
            return model(batch["text"])
        return model(batch["embedding"].to(self.device))

    def _save_artifacts(self, model: FrozenEmbeddingClassifier, metadata: dict[str, Any]) -> None:
        metadata_path = self.output_dir / "metadata.json"
        weights_path = self.output_dir / "best_model.pt"

        torch.save(model.state_dict(), weights_path)
        with metadata_path.open("w", encoding="utf-8") as output_file:
            json.dump(metadata, output_file, indent=2)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_difficulty_class_weights(self, label_statistics: LabelStatistics) -> torch.Tensor:
        counts = torch.tensor(label_statistics.difficulty_counts, dtype=torch.float32, device=self.device)
        counts = torch.clamp(counts, min=1.0)
        weights = label_statistics.num_examples / (len(counts) * counts)
        weights = torch.clamp(weights, max=self.config.max_difficulty_class_weight)
        return weights

    def _build_tag_pos_weights(self, label_statistics: LabelStatistics) -> torch.Tensor:
        positives = torch.tensor(label_statistics.tag_positive_counts, dtype=torch.float32, device=self.device)
        positives = torch.clamp(positives, min=1.0)
        negatives = torch.tensor(label_statistics.num_examples, dtype=torch.float32, device=self.device) - positives
        negatives = torch.clamp(negatives, min=1.0)
        pos_weights = negatives / positives
        pos_weights = torch.clamp(pos_weights, max=self.config.max_tag_pos_weight)
        return pos_weights

    def _compute_difficulty_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.difficulty_loss_type == "weighted_cross_entropy":
            return nn.functional.cross_entropy(logits, labels, weight=class_weights)
        return nn.functional.cross_entropy(logits, labels)

    def _compute_tags_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pos_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.tags_loss_type == "weighted_bce":
            return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weights)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    @staticmethod
    def _compute_macro_f1(
        true_counts: list[int],
        predicted_counts: list[int],
        true_positive_counts: list[int],
    ) -> float:
        per_class_f1 = []
        for true_count, predicted_count, true_positive_count in zip(
            true_counts,
            predicted_counts,
            true_positive_counts,
        ):
            precision = true_positive_count / predicted_count if predicted_count else 0.0
            recall = true_positive_count / true_count if true_count else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_class_f1.append(f1)
        return sum(per_class_f1) / len(per_class_f1) if per_class_f1 else 0.0
