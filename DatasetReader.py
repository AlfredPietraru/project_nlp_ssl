#!/usr/bin/env python3
import csv
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
import shutil
from typing import Iterable, List

logger = logging.getLogger(__name__)


class DatasetEntry:
    def __init__(self, root : Path, problem_id: str, difficulty: str, tags: str) -> None:
        self.id = problem_id
        self.difficulty = difficulty
        self.tags = [tag for tag in tags.split("+") if tag]

        problem_root = root / self.id

        self.description_path = problem_root / "description" / "description.txt"
        if not self.description_path.exists():
            logger.error("Description path not found for problem %s: %s", self.id, self.description_path)

        solutions_dir = problem_root / "solutions_c++"
        if solutions_dir.exists() and solutions_dir.is_dir():
            self.solution_paths = sorted(
                [path for path in solutions_dir.iterdir() if path.is_file()]
            )
        else:
            logger.error("Solutions directory not found for problem %s: %s", self.id, solutions_dir)
            self.solution_paths = []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "description_path": str(self.description_path),
            "solution_paths": [str(path) for path in self.solution_paths],
        }


class DatasetReader:
    def __init__(self, root : Path) -> None:
        self.root = root
        self.json_path = root / "dataset_splits.json"
        self.csv_path =  root / "dataset_splits.csv"
        self.database_path = root / "dataset_splits.db"
        if self.database_path.exists():
            logger.info("Using existing dataset database: %s", self.database_path)
        else:
            self._csv_initializer()
            self._load_csv_into_database()

    def retrieve_train_data(self) -> List[DatasetEntry]:
        return self.retrieve_split_data("train")

    def retrieve_split_data(
        self,
        split_name: str,
        difficulties: Iterable[str] | None = None,
    ) -> List[DatasetEntry]:
        normalized_split = split_name.strip().lower()
        if normalized_split not in {"train", "test"}:
            raise ValueError(f"Unsupported split_name: {split_name}")

        query = f"""
            SELECT * FROM dataset_entries WHERE split = '{normalized_split}'
        """
        rows = self.execute_query(query)

        selected_difficulties = None
        if difficulties is not None:
            selected_difficulties = {
                difficulty.strip().upper()
                for difficulty in difficulties
                if difficulty and difficulty.strip()
            }

        output: list[DatasetEntry] = []
        enriched_count = 0
        enriched_fgh_count = 0
        for row in rows:
            difficulty = str(row["difficulty"]).strip().upper()
            if selected_difficulties is not None and difficulty not in selected_difficulties:
                continue

            entry = DatasetEntry(
                root=self.root,
                problem_id=row["id"],
                difficulty=row["difficulty"],
                tags=row["tags"],
            )
            output.append(entry)

            if "_aug_" in entry.id:
                enriched_count += 1
                if difficulty in {"F", "G", "H"}:
                    enriched_fgh_count += 1

        logger.info(
            "Loaded %s split entries from %s: total=%s enriched=%s enriched_FGH=%s difficulties=%s",
            normalized_split,
            self.root,
            len(output),
            enriched_count,
            enriched_fgh_count,
            sorted(selected_difficulties) if selected_difficulties is not None else "ALL",
        )
        return output

    
    def connect_to_database(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def execute_query(self, query: str):
        connection = self.connect_to_database()
        try:
            cursor = connection.cursor()
            cursor.execute(query)

            if cursor.description is not None:
                rows = cursor.fetchall()
                return [dict(row) for row in rows if row is not None]
            connection.commit()
            return []
        except Exception:
            connection.rollback()
            logger.exception("Failed to execute query against database: %s", self.database_path)
            raise
        finally:
            connection.close()

    def _csv_initializer(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Starting dataset conversion from %s to %s and then into %s",
            self.json_path,
            self.csv_path,
            self.database_path,
        )

        try:
            with self.json_path.open(encoding="utf-8") as input_file:
                data = json.load(input_file)
        except Exception:
            logger.exception("Failed to read json input file: %s", self.json_path)
            return

        total_items_seen = 0
        written_rows = 0
        skipped_items = 0
        split_sizes = {}
        csv_rows = []

        for split in ["train", "test"]:
            split_items = data.get(split, [])
            split_sizes[split] = len(split_items)
            logger.info("Preparing CSV rows for split=%s with %s items", split, len(split_items))

            for item in split_items:
                total_items_seen += 1
                problem_id = item.get("id", "")
                difficulty = item.get("difficulty", "")
                raw_tags = item.get("tags", [])
                normalized_tags = []

                if isinstance(raw_tags, list):
                    for tag in raw_tags:
                        if isinstance(tag, str):
                            cleaned_tag = tag.strip()
                            if cleaned_tag:
                                normalized_tags.append(cleaned_tag)

                if not problem_id or not difficulty:
                    skipped_items += 1
                    logger.warning("Skipping malformed row in split=%s: %s", split, item)
                    continue

                csv_rows.append({
                    "id": problem_id,
                    "difficulty": difficulty,
                    "tags": "+".join(normalized_tags),
                    "split": split,
                })

        try:
            with self.csv_path.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=["id", "difficulty", "tags", "split"],
                )
                writer.writeheader()
                writer.writerows(csv_rows)
                written_rows = len(csv_rows)
        except Exception:
            logger.exception("Failed to write csv output file: %s", self.csv_path)
            return

        logger.info(
            "Finished CSV conversion: seen=%s written=%s skipped=%s split_sizes=%s csv=%s",
            total_items_seen,
            written_rows,
            skipped_items,
            split_sizes,
            self.csv_path,
        )

    def _load_csv_into_database(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Starting sqlite import from %s into %s", self.csv_path, self.database_path)

        try:
            with self.csv_path.open(newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                csv_entries = []
                malformed_rows = 0

                for row in reader:
                    problem_id = (row.get("id") or "").strip()
                    difficulty = (row.get("difficulty") or "").strip()
                    tags = (row.get("tags") or "").strip()
                    split = (row.get("split") or "").strip()

                    if not problem_id or not difficulty or not split:
                        malformed_rows += 1
                        logger.warning("Skipping malformed csv row: %s", row)
                        continue

                    csv_entries.append((problem_id, difficulty, tags, split))
        except Exception:
            logger.exception("Failed to read csv input file: %s", self.csv_path)
            return

        connection = None
        inserted_items = 0
        try:
            connection = self.connect_to_database()
            cursor = connection.cursor()
            cursor.execute("DROP TABLE IF EXISTS dataset_entries")
            cursor.execute(
                """
                CREATE TABLE dataset_entries (
                    id TEXT PRIMARY KEY,
                    difficulty TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    split TEXT NOT NULL
                )
                """
            )
            cursor.executemany(
                """
                INSERT INTO dataset_entries (id, difficulty, tags, split)
                VALUES (?, ?, ?, ?)
                """,
                csv_entries,
            )
            inserted_items = len(csv_entries)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_dataset_entries_split ON dataset_entries(split)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_dataset_entries_difficulty ON dataset_entries(difficulty)"
            )
            connection.commit()
        except Exception:
            logger.exception("Failed to populate sqlite database: %s", self.database_path)
            if connection is not None:
                connection.rollback()
            return
        finally:
            if connection is not None:
                connection.close()

        logger.info(
            "Finished sqlite import: csv_rows=%s inserted=%s malformed_csv_rows=%s database=%s",
            len(csv_entries),
            inserted_items,
            malformed_rows,
            self.database_path,
        )


class EnirchDataset:
    TARGET_DIFFICULTIES = {"F", "G", "H"}
    MAX_BATCH_REWRITE_ATTEMPTS = 2

    def __init__(
        self,
        dataset_reader: DatasetReader,
        output_root: Path,
    ) -> None:
        self.dataset_reader = dataset_reader
        self.source_root = dataset_reader.root
        self.output_root = Path(output_root)
        env_config = self._load_env_file()
        self.api_key = self._get_required_env_value("OPENAI_API_KEY", env_config)
        self.model = self._get_required_env_value("OPENAI_MODEL", env_config)
        self.temperature = self._get_float_env_value("LLM_TEMPERATURE", env_config)
        self.max_tokens = self._get_int_env_value("LLM_MAX_TOKENS", env_config)
        self.client = self._create_openai_client()

    def clone_and_enrich(
        self,
        difficulties: Iterable[str] | None = None,
        split_names: Iterable[str] | None = None,
        overwrite: bool = False,
        batch_size: int = 10,
    ) -> dict[str, int]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        selected_difficulties = {
            difficulty.strip().upper()
            for difficulty in (difficulties or self.TARGET_DIFFICULTIES)
            if difficulty and difficulty.strip()
        }
        selected_splits = {split.strip() for split in split_names} if split_names else None

        self._clone_dataset(overwrite=overwrite)
        metadata_path = self.output_root / "dataset_splits.json"
        metadata = self._load_metadata(metadata_path)
        self._invalidate_cached_indexes()

        pending_tasks: list[dict] = []
        for split_name, items in metadata.items():
            if not isinstance(items, list):
                continue
            if selected_splits is not None and split_name not in selected_splits:
                continue

            for item in list(items):
                difficulty = str(item.get("difficulty", "")).strip().upper()
                if difficulty not in selected_difficulties:
                    continue

                source_problem_id = item.get("id")
                if not source_problem_id:
                    logger.warning("Skipping metadata row without id in split=%s: %s", split_name, item)
                    continue

                augmented_problem_id = self._build_augmented_problem_id(source_problem_id)
                pending_tasks.append(
                    {
                        "split_name": split_name,
                        "source_problem_id": source_problem_id,
                        "augmented_problem_id": augmented_problem_id,
                        "difficulty": difficulty,
                        "tags": item.get("tags", []),
                        "metadata_item": dict(item),
                    }
                )

        total_tasks = len(pending_tasks)
        logger.info(
            "Enrichment starting: %s problem descriptions will be converted in batches of %s",
            total_tasks,
            batch_size,
        )
        if total_tasks == 0:
            logger.info("No matching problems found for enrichment. Metadata remains unchanged.")

        skipped_problem_ids: list[str] = []
        created_entries = 0
        for start_index in range(0, len(pending_tasks), batch_size):
            batch_tasks = pending_tasks[start_index:start_index + batch_size]
            batch_number = (start_index // batch_size) + 1
            total_batches = (len(pending_tasks) + batch_size - 1) // batch_size

            logger.info(
                "Submitting description batch %s/%s covering problems %s-%s of %s",
                batch_number,
                total_batches,
                start_index + 1,
                start_index + len(batch_tasks),
                len(pending_tasks),
            )

            batch_items = []
            for task in batch_tasks:
                description_path = (
                    self.output_root
                    / task["source_problem_id"]
                    / "description"
                    / "description.txt"
                )
                with description_path.open(encoding="utf-8", errors="ignore") as input_file:
                    batch_items.append(
                        {
                            "problem_id": task["source_problem_id"],
                            "difficulty": task["difficulty"],
                            "tags": task["tags"] or [],
                            "description": input_file.read().strip(),
                        }
                    )

            rewritten_descriptions = self._rewrite_problem_descriptions_batch(batch_items)
            task_by_problem_id = {
                task["source_problem_id"]: task
                for task in batch_tasks
            }

            for batch_offset, task in enumerate(batch_tasks, start=1):
                rewritten_description = rewritten_descriptions.get(task["source_problem_id"])
                absolute_index = start_index + batch_offset
                if not rewritten_description:
                    skipped_problem_ids.append(task["source_problem_id"])
                    logger.warning(
                        "Skipping augmented copy for %s because no rewritten description was produced",
                        task["source_problem_id"],
                    )
                    continue

                self._persist_augmented_problem(
                    task=task_by_problem_id[task["source_problem_id"]],
                    rewritten_description=rewritten_description,
                    metadata=metadata,
                    metadata_path=metadata_path,
                )
                created_entries += 1
                logger.info(
                    "Created augmented problem %s from %s. Progress: %s/%s done, %s left",
                    task["augmented_problem_id"],
                    task["source_problem_id"],
                    absolute_index,
                    total_tasks,
                    total_tasks - absolute_index,
                )

            logger.info(
                "Completed description batch %s/%s. Persisted %s items from this batch.",
                batch_number,
                total_batches,
                sum(
                    1
                    for task in batch_tasks
                    if task["source_problem_id"] in rewritten_descriptions
                ),
            )

        DatasetReader(self.output_root)
        if skipped_problem_ids:
            logger.warning(
                "Finished enrichment with %s skipped problems: %s",
                len(skipped_problem_ids),
                ", ".join(skipped_problem_ids),
            )

        return {
            "created_entries": created_entries,
            "targeted_difficulties": len(selected_difficulties),
        }

    def _clone_dataset(self, overwrite: bool) -> None:
        if self.output_root.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output dataset already exists: {self.output_root}. "
                    "Pass overwrite=True to replace it."
                )
            shutil.rmtree(self.output_root)

        shutil.copytree(self.source_root, self.output_root)
        logger.info("Cloned dataset from %s to %s", self.source_root, self.output_root)

    def _load_metadata(self, metadata_path: Path) -> dict:
        with metadata_path.open(encoding="utf-8") as input_file:
            return json.load(input_file)

    def _write_metadata(self, metadata_path: Path, metadata: dict) -> None:
        with metadata_path.open("w", encoding="utf-8") as output_file:
            json.dump(metadata, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")

    def _build_augmented_problem_id(self, source_problem_id: str) -> str:
        suffix = 1
        while True:
            candidate = f"{source_problem_id}_aug_{suffix}"
            if not (self.output_root / candidate).exists():
                return candidate
            suffix += 1

    def _create_augmented_problem(
        self,
        source_problem_id: str,
        augmented_problem_id: str,
        rewritten_description: str,
    ) -> None:
        source_problem_root = self.output_root / source_problem_id
        target_problem_root = self.output_root / augmented_problem_id
        if target_problem_root.exists():
            raise FileExistsError(f"Augmented problem already exists: {target_problem_root}")

        description_path = source_problem_root / "description" / "description.txt"
        solutions_dir = source_problem_root / "solutions_c++"
        if not description_path.exists() or not solutions_dir.exists():
            raise FileNotFoundError(
                f"Problem {source_problem_id} is missing description or solutions in {source_problem_root}"
            )

        shutil.copytree(source_problem_root, target_problem_root)

        target_description_path = target_problem_root / "description" / "description.txt"
        with target_description_path.open("w", encoding="utf-8") as output_file:
            output_file.write(rewritten_description)

        logger.info("Created augmented copy %s from %s", augmented_problem_id, source_problem_id)

    def _persist_augmented_problem(
        self,
        task: dict,
        rewritten_description: str,
        metadata: dict,
        metadata_path: Path,
    ) -> None:
        self._create_augmented_problem(
            source_problem_id=task["source_problem_id"],
            augmented_problem_id=task["augmented_problem_id"],
            rewritten_description=rewritten_description,
        )

        new_item = dict(task["metadata_item"])
        new_item["id"] = task["augmented_problem_id"]
        metadata[task["split_name"]].append(new_item)
        self._write_metadata(metadata_path, metadata)

    def _rewrite_problem_description(
        self,
        problem_id: str,
        difficulty: str,
        tags: list[str],
        description: str,
    ) -> str:
        payload = {
            "task": "rewrite_problem_description",
            "problem_id": problem_id,
            "difficulty": difficulty,
            "tags": tags,
            "description": description,
        }
        instructions = (
            "You are rewriting competitive programming problem statements. "
            "Preserve the exact task, constraints, required outputs, and algorithmic intent. "
            "Only rephrase wording and presentation. Return plain text only."
        )
        return self._generate_text(instructions=instructions, payload=payload)

    def _rewrite_problem_descriptions_batch(
        self,
        batch_items: list[dict],
    ) -> dict[str, str]:
        if not batch_items:
            return {}

        payload = {
            "task": "rewrite_problem_descriptions_batch",
            "problems": batch_items,
        }
        instructions = (
            "You are rewriting multiple competitive programming problem statements. "
            "Preserve the exact task, constraints, required outputs, and algorithmic intent for each problem. "
            "Only rephrase wording and presentation. Return a valid JSON object only. "
            "Use each provided problem_id as the key and the rewritten plain-text description as the value. "
            "Do not omit any problem and do not wrap the JSON in markdown."
        )

        for attempt in range(1, self.MAX_BATCH_REWRITE_ATTEMPTS + 1):
            missing_problem_ids: list[str] = []
            parsed_output: dict | None = None

            try:
                output_text = self._generate_text(instructions=instructions, payload=payload)
                parsed_output = self._parse_batch_rewrite_output(output_text)
                if not isinstance(parsed_output, dict):
                    raise ValueError("Batch description rewrite did not return a JSON object")
            except Exception:
                logger.exception(
                    "Batch description rewrite failed on attempt %s/%s for problems: %s",
                    attempt,
                    self.MAX_BATCH_REWRITE_ATTEMPTS,
                    ", ".join(item["problem_id"] for item in batch_items),
                )
                if attempt == self.MAX_BATCH_REWRITE_ATTEMPTS:
                    return self._rewrite_problem_descriptions_individually(batch_items)
                continue

            rewritten_descriptions: dict[str, str] = {}
            for item in batch_items:
                problem_id = item["problem_id"]
                rewritten_description = parsed_output.get(problem_id)
                if not isinstance(rewritten_description, str) or not rewritten_description.strip():
                    missing_problem_ids.append(problem_id)
                    continue
                rewritten_descriptions[problem_id] = rewritten_description.strip()

            if not missing_problem_ids:
                return rewritten_descriptions

            logger.warning(
                "Batch description rewrite returned %s/%s items; retrying missing problems individually: %s",
                len(rewritten_descriptions),
                len(batch_items),
                ", ".join(missing_problem_ids),
            )
            missing_batch_items = [
                item for item in batch_items
                if item["problem_id"] in set(missing_problem_ids)
            ]
            rewritten_descriptions.update(
                self._rewrite_problem_descriptions_individually(missing_batch_items)
            )
            return rewritten_descriptions

        return {}

    def _parse_batch_rewrite_output(self, output_text: str) -> dict:
        normalized_output = self._strip_json_code_fences(output_text)
        try:
            return json.loads(normalized_output)
        except json.JSONDecodeError as exc:
            repaired_output = self._escape_invalid_json_backslashes(normalized_output)
            if repaired_output != normalized_output:
                logger.warning("Repairing malformed JSON returned by batch rewrite: %s", exc)
                return json.loads(repaired_output)
            raise

    def _strip_json_code_fences(self, text: str) -> str:
        stripped = text.strip()
        fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1).strip()
        return stripped

    def _escape_invalid_json_backslashes(self, text: str) -> str:
        valid_escape_chars = {'"', "\\", "/", "b", "f", "n", "r", "t"}
        repaired_chars: list[str] = []
        in_string = False
        index = 0

        while index < len(text):
            char = text[index]

            if char == '"' and not self._is_escaped(text, index):
                in_string = not in_string
                repaired_chars.append(char)
                index += 1
                continue

            if in_string and char == "\\":
                next_index = index + 1
                if next_index >= len(text):
                    repaired_chars.append("\\\\")
                    index += 1
                    continue

                next_char = text[next_index]
                if next_char in valid_escape_chars:
                    repaired_chars.append(char)
                    repaired_chars.append(next_char)
                    index += 2
                    continue

                if next_char == "u" and next_index + 4 < len(text):
                    hex_digits = text[next_index + 1:next_index + 5]
                    if all(digit in "0123456789abcdefABCDEF" for digit in hex_digits):
                        repaired_chars.append(char)
                        repaired_chars.append("u")
                        repaired_chars.extend(hex_digits)
                        index += 6
                        continue

                repaired_chars.append("\\\\")
                index += 1
                continue

            repaired_chars.append(char)
            index += 1

        return "".join(repaired_chars)

    def _is_escaped(self, text: str, index: int) -> bool:
        backslash_count = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslash_count += 1
            cursor -= 1
        return backslash_count % 2 == 1

    def _rewrite_problem_descriptions_individually(
        self,
        batch_items: list[dict],
    ) -> dict[str, str]:
        rewritten_descriptions: dict[str, str] = {}
        for item in batch_items:
            problem_id = item["problem_id"]
            try:
                rewritten_description = self._rewrite_problem_description(
                    problem_id=problem_id,
                    difficulty=item["difficulty"],
                    tags=item["tags"],
                    description=item["description"],
                ).strip()
            except Exception:
                logger.exception("Individual description rewrite failed for problem %s", problem_id)
                continue

            if not rewritten_description:
                logger.warning("Individual description rewrite returned empty text for problem %s", problem_id)
                continue

            rewritten_descriptions[problem_id] = rewritten_description

        return rewritten_descriptions

    def _generate_text(self, instructions: str, payload: dict) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        output_text = getattr(response, "output_text", "").strip()
        if not output_text:
            raise ValueError("OpenAI response did not contain output_text")
        return output_text

    def _create_openai_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("The openai package is required to enrich the dataset") from exc

        return OpenAI(api_key=self.api_key)

    def _load_env_file(self) -> dict[str, str]:
        env_path = Path(".env")
        if not env_path.exists():
            raise FileNotFoundError(f".env file not found: {env_path.resolve()}")

        env_config: dict[str, str] = {}
        with env_path.open(encoding="utf-8") as input_file:
            for raw_line in input_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env_config[key.strip()] = value.strip().strip("'\"")
        return env_config

    def _get_required_env_value(self, key: str, env_config: dict[str, str]) -> str:
        value = os.environ.get(key) or env_config.get(key)
        if not value:
            raise EnvironmentError(f"{key} is not set in the environment or .env file")
        return value

    def _get_float_env_value(self, key: str, env_config: dict[str, str]) -> float:
        value = self._get_required_env_value(key, env_config)
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{key} must be a float, got: {value}") from exc

    def _get_int_env_value(self, key: str, env_config: dict[str, str]) -> int:
        value = self._get_required_env_value(key, env_config)
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer, got: {value}") from exc

    def _invalidate_cached_indexes(self) -> None:
        for cache_path in (
            self.output_root / "dataset_splits.csv",
            self.output_root / "dataset_splits.db",
        ):
            if cache_path.exists():
                cache_path.unlink()
                logger.info("Removed stale cache file: %s", cache_path)




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dataset_reader = DatasetReader(Path("cleaned_data"))
    # data = dataset_reader.retrieve_train_data()
    # print(json.dumps([entry.to_dict() for entry in data], indent=2))
    enricher = EnirchDataset(dataset_reader=dataset_reader, output_root=Path("augmented_data"))
    enricher.clone_and_enrich(batch_size=10, overwrite=True)
