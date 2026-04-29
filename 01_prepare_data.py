"""
Prepare training data for LLM fine-tuning.
Scans all problems in cleaned_data and builds train/validation/test jsonl files.
"""
import json
import random
from collections import defaultdict
from pathlib import Path


BASE_PATH = Path("/home/alf/nlp_ssl/cleaned_data")
OUTPUT_DIR = Path("/home/alf/nlp_ssl/training_data")
SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
RANDOM_SEED = 42


def load_metadata():
    """Load optional metadata from dataset_splits.json keyed by problem id."""
    metadata_path = BASE_PATH / "dataset_splits.json"
    metadata = {}

    if not metadata_path.exists():
        print("⚠ dataset_splits.json not found, continuing with folder-derived metadata only")
        return metadata

    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)

    for split_name in ("train", "validation", "test"):
        for problem_info in data.get(split_name, []):
            problem_id = problem_info.get("id")
            if not problem_id:
                continue
            metadata[problem_id] = {
                "difficulty": problem_info.get("difficulty"),
                "tags": problem_info.get("tags", []),
            }

    return metadata


def iter_problem_dirs():
    """Yield all valid problem directories from cleaned_data."""
    for path in sorted(BASE_PATH.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "description").exists():
            continue
        yield path


def infer_difficulty(problem_id, metadata):
    """Use metadata first, then derive difficulty from the folder suffix."""
    difficulty = metadata.get("difficulty")
    if difficulty:
        return difficulty

    if "_" in problem_id:
        return problem_id.rsplit("_", 1)[-1]

    return None


def choose_solution_file(solution_dir):
    """Select a deterministic solution file from the available solutions."""
    if not solution_dir.exists():
        return None

    solutions = sorted(solution_dir.glob("*.txt"))
    return solutions[0] if solutions else None


def build_examples():
    """Read every problem directory and convert it to a training example."""
    metadata_map = load_metadata()
    examples = []

    for problem_dir in iter_problem_dirs():
        problem_id = problem_dir.name
        metadata = metadata_map.get(problem_id, {})
        difficulty = infer_difficulty(problem_id, metadata)
        tags = metadata.get("tags", [])

        desc_file = problem_dir / "description" / "description.txt"
        solution_file = choose_solution_file(problem_dir / "solutions_c++")

        if not desc_file.exists() or not solution_file:
            print(f"⚠ Skipping {problem_id}: missing description or solution")
            continue

        try:
            with open(desc_file, encoding="utf-8", errors="ignore") as f:
                description = f.read().strip()

            with open(solution_file, encoding="utf-8", errors="ignore") as f:
                solution = f.read().strip()
        except Exception as exc:
            print(f"✗ Error processing {problem_id}: {exc}")
            continue

        if not description or not solution:
            print(f"⚠ Skipping {problem_id}: empty description or solution")
            continue

        if not difficulty:
            print(f"⚠ Skipping {problem_id}: missing difficulty")
            continue

        examples.append({
            "id": problem_id,
            "text": f"{description}\n\n### Solution ###\n{solution}",
            "difficulty": difficulty,
            "tags": tags,
        })
        print(f"✓ {problem_id}: difficulty={difficulty}, tags={len(tags)}")

    return examples


def split_examples(examples):
    """Create deterministic train/validation/test splits from all examples."""
    shuffled = list(examples)
    random.Random(RANDOM_SEED).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * SPLIT_RATIOS["train"])
    validation_end = train_end + int(total * SPLIT_RATIOS["validation"])

    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def save_splits(splits):
    OUTPUT_DIR.mkdir(exist_ok=True)

    for split_name, examples in splits.items():
        output_file = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for ex in examples:
                json.dump(ex, f, ensure_ascii=False)
                f.write("\n")
        print(f"\n✓ Saved {output_file} ({len(examples)} examples)")


def print_statistics(splits):
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)

    total = sum(len(v) for v in splits.values())
    for split_name, examples in splits.items():
        pct = 100 * len(examples) / total if total else 0
        print(f"{split_name:12} : {len(examples):4} examples ({pct:.1f}%)")
    print(f"{'TOTAL':12} : {total:4} examples")

    all_tags = defaultdict(int)
    for split_examples in splits.values():
        for ex in split_examples:
            for tag in ex["tags"]:
                all_tags[tag] += 1

    print("\nTop 15 Tags:")
    print("-" * 60)
    for tag, count in sorted(all_tags.items(), key=lambda x: (-x[1], x[0]))[:15]:
        print(f"  {tag:30} : {count:4} problems")

    difficulty_dist = defaultdict(int)
    for split_examples in splits.values():
        for ex in split_examples:
            difficulty_dist[ex["difficulty"]] += 1

    print("\nDifficulty Distribution:")
    print("-" * 60)
    for diff in sorted(difficulty_dist):
        print(f"  {diff} : {difficulty_dist[diff]:3} problems")


def prepare_training_data():
    """Main entrypoint for preparing all problem data."""
    examples = build_examples()
    splits = split_examples(examples)
    print_statistics(splits)
    save_splits(splits)
    return splits


if __name__ == "__main__":
    print("🚀 Starting data preparation...\n")
    prepare_training_data()
    print("\n✅ Data preparation complete!")
