from __future__ import annotations

import random
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from DatasetReader import DatasetEntry, DatasetReader


class RunningMode(str, Enum):
    DESCRIPTION_ONLY = "DESCRIPTION_ONLY"
    SOLUTION_ONLY = "SOLUTION_ONLY"
    DESCRIPTION_AND_SOLUTION = "DESCRIPTION_AND_SOLUTION"


class TextEmbeddingModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = self._mean_pool(
            token_embeddings=outputs.last_hidden_state,
            attention_mask=inputs["attention_mask"],
        )
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0).cpu()

    @staticmethod
    def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed_embeddings = torch.sum(token_embeddings * expanded_mask, dim=1)
        summed_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        return summed_embeddings / summed_mask


class ProgrammingProblemsDataset(Dataset):
    def __init__(
        self,
        dataset_reader: DatasetReader,
        is_train: bool,
        running_mode: RunningMode,
        embedding_model: TextEmbeddingModel | None = None,
        include_embeddings: bool = False,
        entries: list[DatasetEntry] | None = None,
        sample_random_solution: bool | None = None,
    ) -> None:
        self.dataset_reader = dataset_reader
        self.is_train = is_train
        self.running_mode = running_mode
        self.embedding_model = embedding_model
        self.include_embeddings = include_embeddings
        self.entries = entries if entries is not None else self._load_entries()
        self.sample_random_solution = is_train if sample_random_solution is None else sample_random_solution
        self._embedding_cache: dict[tuple[str, str], torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        description_text = self._read_text(entry.description_path)
        solution_text = self._select_solution_text(entry)

        item = {
            "id": entry.id,
            "difficulty": entry.difficulty,
            "tags": entry.tags,
            "description": description_text,
            "solution": solution_text,
            "text": self._build_text(description_text, solution_text),
        }

        if self.include_embeddings:
            item["embedding"] = self._get_embedding(entry.id, item["text"])

        return item

    def _load_entries(self) -> list[DatasetEntry]:
        split_name = "train" if self.is_train else "test"
        return self.dataset_reader.retrieve_split_data(split_name)

    def _select_solution_text(self, entry: DatasetEntry) -> str:
        if not entry.solution_paths:
            return ""

        if self.sample_random_solution:
            solution_path = random.choice(entry.solution_paths)
        else:
            solution_path = entry.solution_paths[0]

        return self._read_text(solution_path)

    def _build_text(self, description_text: str, solution_text: str) -> str:
        if self.running_mode == RunningMode.DESCRIPTION_ONLY:
            return description_text

        if self.running_mode == RunningMode.SOLUTION_ONLY:
            return solution_text

        parts = []
        if description_text:
            parts.append("Description:\n" + description_text)
        if solution_text:
            parts.append("Solution:\n" + solution_text)
        return "\n\n".join(parts)

    def _get_embedding(self, problem_id: str, text: str) -> torch.Tensor:
        if self.embedding_model is None:
            raise ValueError("include_embeddings=True requires an embedding_model instance")

        cache_key = (problem_id, text)
        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = self.embedding_model.encode(text)

        return self._embedding_cache[cache_key]

    @staticmethod
    def _read_text(path: Path) -> str:
        if not path.exists():
            return ""

        with path.open(encoding="utf-8", errors="ignore") as input_file:
            return input_file.read().strip()


class ProgrammingProblemsDataLoader(DataLoader):
    def __init__(
        self,
        dataset: ProgrammingProblemsDataset,
        batch_size: int = 8,
        shuffle: bool | None = None,
        num_workers: int = 0,
    ) -> None:
        if shuffle is None:
            shuffle = dataset.is_train

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated = {
            "id": [item["id"] for item in batch],
            "difficulty": [item["difficulty"] for item in batch],
            "tags": [item["tags"] for item in batch],
            "description": [item["description"] for item in batch],
            "solution": [item["solution"] for item in batch],
            "text": [item["text"] for item in batch],
        }

        if "embedding" in batch[0]:
            collated["embedding"] = torch.stack([item["embedding"] for item in batch])

        return collated
