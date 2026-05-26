#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt

from DatasetReader import DatasetReader


DIFFICULTY_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]


class DatasetPlotter:
    def __init__(self, dataset_reader: DatasetReader, output_dir: str | Path = "plots") -> None:
        self.dataset_reader = dataset_reader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_train_entries(self):
        return self.dataset_reader.retrieve_train_data()

    def _save_bar_chart(
        self,
        labels: list[str],
        values: list[int],
        title: str,
        ylabel: str,
        output_path: Path,
        rotate_labels: bool = False,
    ) -> None:
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color="#4C72B0", edgecolor="black")
        plt.title(title)
        plt.ylabel(ylabel)
        if rotate_labels:
            plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close()

    def _save_json(self, data: dict[str, int], output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(data, output_file, indent=2)

    def plot_difficulty_distribution(self, output_filename: str = "train_difficulty_distribution.png") -> Path:
        entries = self._load_train_entries()
        difficulty_counter = Counter(entry.difficulty for entry in entries if entry.difficulty)

        labels = [difficulty for difficulty in DIFFICULTY_ORDER if difficulty in difficulty_counter]
        values = [difficulty_counter[difficulty] for difficulty in labels]

        output_path = self.output_dir / output_filename
        self._save_bar_chart(
            labels=labels,
            values=values,
            title="Train Difficulty Distribution",
            ylabel="Count",
            output_path=output_path,
        )
        return output_path

    def save_difficulty_distribution_json(
        self,
        output_filename: str = "train_difficulty_distribution.json",
    ) -> Path:
        entries = self._load_train_entries()
        difficulty_counter = Counter(entry.difficulty for entry in entries if entry.difficulty)

        distribution = {
            difficulty: difficulty_counter[difficulty]
            for difficulty in DIFFICULTY_ORDER
            if difficulty in difficulty_counter
        }
        output_path = self.output_dir / output_filename
        self._save_json(distribution, output_path)
        return output_path

    def plot_label_distribution(self, output_filename: str = "train_label_distribution.png") -> Path:
        entries = self._load_train_entries()
        label_counter = Counter()
        for entry in entries:
            label_counter.update(entry.tags)

        labels = [label for label, _ in label_counter.most_common()]
        values = [label_counter[label] for label in labels]

        output_path = self.output_dir / output_filename
        self._save_bar_chart(
            labels=labels,
            values=values,
            title="Train Label Distribution",
            ylabel="Count",
            output_path=output_path,
            rotate_labels=True,
        )
        return output_path

    def save_label_distribution_json(
        self,
        output_filename: str = "train_label_distribution.json",
    ) -> Path:
        entries = self._load_train_entries()
        label_counter = Counter()
        for entry in entries:
            label_counter.update(entry.tags)

        distribution = {
            label: count for label, count in label_counter.most_common()
        }
        output_path = self.output_dir / output_filename
        self._save_json(distribution, output_path)
        return output_path

    def plot_all(self) -> dict[str, Path]:
        return {
            "difficulty_distribution": self.plot_difficulty_distribution(),
            "difficulty_distribution_json": self.save_difficulty_distribution_json(),
            "label_distribution": self.plot_label_distribution(),
            "label_distribution_json": self.save_label_distribution_json(),
        }


if __name__ == "__main__":
    plotter = DatasetPlotter(dataset_reader=DatasetReader(Path("augmented_data")), 
                             output_dir=Path("augmented_plots"))
    output_paths = plotter.plot_all()
    for plot_name, output_path in output_paths.items():
        print(f"{plot_name}: {output_path}")
