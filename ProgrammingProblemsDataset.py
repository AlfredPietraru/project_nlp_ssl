from __future__ import annotations

import random
from enum import Enum
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset

from DatasetReader import DatasetEntry, DatasetReader


class RunningMode(str, Enum):
    DESCRIPTION_ONLY = "DESCRIPTION_ONLY"
    SOLUTION_ONLY = "SOLUTION_ONLY"
    DESCRIPTION_AND_SOLUTION = "DESCRIPTION_AND_SOLUTION"


class ProgrammingProblemsDataset(Dataset):
    def __init__(
        self,
        dataset_reader: DatasetReader,
        is_train: bool,
        running_mode: RunningMode,
    ) -> None:
        self.dataset_reader = dataset_reader
        self.is_train = is_train
        self.running_mode = running_mode
        self.entries = self._load_entries()

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

        return item

    def _load_entries(self) -> list[DatasetEntry]:
        split_name = "train" if self.is_train else "test"
        return self.dataset_reader.retrieve_split_data(split_name)

    def _select_solution_text(self, entry: DatasetEntry) -> str:
        if not entry.solution_paths:
            return ""

        if self.is_train:
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
    def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
        return {
            "id": [item["id"] for item in batch],
            "difficulty": [item["difficulty"] for item in batch],
            "tags": [item["tags"] for item in batch],
            "description": [item["description"] for item in batch],
            "solution": [item["solution"] for item in batch],
            "text": [item["text"] for item in batch],
        }
