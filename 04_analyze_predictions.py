"""
Generate summary results and graphics from predictions.json.

Outputs:
- summary.json with aggregate metrics
- per_problem_comparison.jsonl with prediction vs ground truth
- predicted, ground-truth, and training-set difficulty distribution charts
- predicted and ground-truth tag distribution charts for the evaluated set
- confusion-matrix artifacts for difficulty and tags
- Optional evaluation metrics if ground truth metadata is available
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        multilabel_confusion_matrix,
        precision_score,
        recall_score,
    )
except Exception:  # pragma: no cover
    accuracy_score = None
    confusion_matrix = None
    f1_score = None
    multilabel_confusion_matrix = None
    precision_score = None
    recall_score = None


DIFFICULTY_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth(metadata_path):
    """Load ground truth keyed by problem id from dataset_splits.json."""
    if not metadata_path.exists():
        return {}

    data = load_json(metadata_path)
    truth = {}
    for split_name in ("train", "validation", "test"):
        for item in data.get(split_name, []):
            problem_id = item.get("id")
            if not problem_id:
                continue
            truth[problem_id] = {
                "difficulty": item.get("difficulty"),
                "tags": item.get("tags", []),
                "split": split_name,
            }
    return truth


def load_training_difficulty_distribution(training_path):
    """Load difficulty counts from training_data/train.jsonl if available."""
    training_file = Path(training_path)
    if not training_file.exists():
        return Counter()

    difficulty_counter = Counter()
    with open(training_file, encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            difficulty = example.get("difficulty")
            if difficulty:
                difficulty_counter[difficulty] += 1
    return difficulty_counter


def ensure_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def save_bar_chart(labels, values, title, ylabel, output_path, rotate=False):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    if rotate:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_grouped_bar_chart(labels, left_values, right_values, left_name, right_name, title, ylabel, output_path):
    positions = range(len(labels))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar([p - width / 2 for p in positions], left_values, width=width, label=left_name)
    plt.bar([p + width / 2 for p in positions], right_values, width=width, label=right_name)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(list(positions), labels, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_histogram(values, bins, title, xlabel, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_confusion_matrix(y_true, y_pred, labels, output_path):
    if confusion_matrix is None or not y_true:
        return

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Difficulty Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_tag_confusion_heatmap(tag_rows, output_path, top_k=20):
    """Save a compact heatmap of per-tag TP/FP/FN/TN counts."""
    if not tag_rows:
        return

    rows = sorted(tag_rows, key=lambda row: row["support"], reverse=True)[:top_k]
    metrics = ["tp", "fp", "fn", "tn"]
    matrix = [[row[metric] for metric in metrics] for row in rows]
    labels = [row["tag"] for row in rows]

    plt.figure(figsize=(9, max(6, len(rows) * 0.35)))
    plt.imshow(matrix, cmap="Blues", aspect="auto")
    plt.title(f"Tag Confusion Counts (Top {len(rows)} by support)")
    plt.xlabel("Metric")
    plt.ylabel("Tag")
    plt.xticks(range(len(metrics)), [metric.upper() for metric in metrics])
    plt.yticks(range(len(labels)), labels)

    for i in range(len(rows)):
        for j in range(len(metrics)):
            plt.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def compute_tag_metrics(predicted_tags, true_tags):
    pred_set = set(predicted_tags)
    true_set = set(true_tags)

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    jaccard = tp / len(pred_set | true_set) if (pred_set | true_set) else 1.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }


def build_tag_vectors(predicted_tags, true_tags, all_tags):
    pred_set = set(predicted_tags)
    true_set = set(true_tags)
    true_vec = [1 if tag in true_set else 0 for tag in all_tags]
    pred_vec = [1 if tag in pred_set else 0 for tag in all_tags]
    return true_vec, pred_vec


def analyze_predictions(predictions, ground_truth):
    difficulty_counter = Counter()
    true_difficulty_counter = Counter()
    tag_counter = Counter()
    true_tag_counter = Counter()
    confidence_values = []
    tag_count_values = []

    evaluated_ids = []
    difficulty_true = []
    difficulty_pred = []
    tag_metric_rows = []
    comparison_rows = []
    tag_true_vectors = []
    tag_pred_vectors = []
    evaluation_tags = sorted({tag for item in ground_truth.values() for tag in item.get("tags", [])})
    total_tag_tp = 0
    total_tag_fp = 0
    total_tag_fn = 0

    for problem_id, pred in predictions.items():
        difficulty = pred.get("difficulty", "Unknown")
        difficulty_counter[difficulty] += 1

        confidence = pred.get("difficulty_confidence")
        if isinstance(confidence, (int, float)):
            confidence_values.append(confidence)

        predicted_tags = pred.get("tags", [])
        tag_count_values.append(len(predicted_tags))
        tag_counter.update(predicted_tags)

        if problem_id in ground_truth:
            truth = ground_truth[problem_id]
            evaluated_ids.append(problem_id)
            true_difficulty = truth.get("difficulty")
            true_tags = truth.get("tags", [])

            difficulty_true.append(true_difficulty)
            difficulty_pred.append(difficulty)
            true_difficulty_counter[true_difficulty] += 1
            true_tag_counter.update(true_tags)

            tag_metrics = compute_tag_metrics(predicted_tags, true_tags)
            tag_metric_rows.append(tag_metrics)
            true_vec, pred_vec = build_tag_vectors(predicted_tags, true_tags, evaluation_tags)
            tag_true_vectors.append(true_vec)
            tag_pred_vectors.append(pred_vec)
            total_tag_tp += sum(1 for tag in predicted_tags if tag in set(true_tags))
            total_tag_fp += sum(1 for tag in predicted_tags if tag not in set(true_tags))
            total_tag_fn += sum(1 for tag in true_tags if tag not in set(predicted_tags))
            comparison_rows.append({
                "id": problem_id,
                "split": truth.get("split"),
                "true_difficulty": true_difficulty,
                "predicted_difficulty": difficulty,
                "difficulty_correct": true_difficulty == difficulty,
                "difficulty_confidence": pred.get("difficulty_confidence"),
                "true_tags": true_tags,
                "predicted_tags": predicted_tags,
                "missing_tags": sorted(set(true_tags) - set(predicted_tags)),
                "extra_tags": sorted(set(predicted_tags) - set(true_tags)),
                "tag_precision": tag_metrics["precision"],
                "tag_recall": tag_metrics["recall"],
                "tag_f1": tag_metrics["f1"],
                "tag_jaccard": tag_metrics["jaccard"],
            })

    summary = {
        "num_predictions": len(predictions),
        "num_ground_truth_items": len(ground_truth),
        "predicted_difficulty_distribution": {
            diff: difficulty_counter.get(diff, 0) for diff in DIFFICULTY_ORDER if difficulty_counter.get(diff, 0) > 0
        },
        "top_predicted_tags": dict(tag_counter.most_common(15)),
        "avg_difficulty_confidence": (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        ),
        "avg_predicted_tags_per_problem": (
            sum(tag_count_values) / len(tag_count_values) if tag_count_values else 0.0
        ),
    }

    if evaluated_ids:
        correct = sum(int(t == p) for t, p in zip(difficulty_true, difficulty_pred))
        difficulty_accuracy = correct / len(evaluated_ids)
        difficulty_macro_f1 = None
        difficulty_weighted_f1 = None
        tag_micro_precision = None
        tag_micro_recall = None
        tag_micro_f1 = None
        tag_macro_precision = None
        tag_macro_recall = None
        tag_macro_f1 = None
        difficulty_per_class = []
        tag_per_class = []

        if accuracy_score is not None and f1_score is not None:
            difficulty_accuracy = accuracy_score(difficulty_true, difficulty_pred)
            difficulty_macro_f1 = f1_score(
                difficulty_true,
                difficulty_pred,
                labels=DIFFICULTY_ORDER,
                average="macro",
                zero_division=0,
            )
            difficulty_weighted_f1 = f1_score(
                difficulty_true,
                difficulty_pred,
                labels=DIFFICULTY_ORDER,
                average="weighted",
                zero_division=0,
            )

            for difficulty_label in DIFFICULTY_ORDER:
                support = sum(1 for value in difficulty_true if value == difficulty_label)
                if support == 0:
                    continue
                class_correct = sum(
                    1 for true_value, pred_value in zip(difficulty_true, difficulty_pred)
                    if true_value == difficulty_label and pred_value == difficulty_label
                )
                class_accuracy = class_correct / support if support else 0.0
                class_f1 = f1_score(
                    difficulty_true,
                    difficulty_pred,
                    labels=[difficulty_label],
                    average="macro",
                    zero_division=0,
                )
                difficulty_per_class.append({
                    "difficulty": difficulty_label,
                    "support": support,
                    "correct": class_correct,
                    "accuracy": class_accuracy,
                    "f1": class_f1,
                })

        if tag_true_vectors and precision_score is not None and recall_score is not None and f1_score is not None:
            tag_micro_precision = precision_score(
                tag_true_vectors,
                tag_pred_vectors,
                average="micro",
                zero_division=0,
            )
            tag_micro_recall = recall_score(
                tag_true_vectors,
                tag_pred_vectors,
                average="micro",
                zero_division=0,
            )
            tag_micro_f1 = f1_score(
                tag_true_vectors,
                tag_pred_vectors,
                average="micro",
                zero_division=0,
            )
            tag_macro_precision = precision_score(
                tag_true_vectors,
                tag_pred_vectors,
                average="macro",
                zero_division=0,
            )
            tag_macro_recall = recall_score(
                tag_true_vectors,
                tag_pred_vectors,
                average="macro",
                zero_division=0,
            )
            tag_macro_f1 = f1_score(
                tag_true_vectors,
                tag_pred_vectors,
                average="macro",
                zero_division=0,
            )

            for index, tag_name in enumerate(evaluation_tags):
                true_column = [row[index] for row in tag_true_vectors]
                pred_column = [row[index] for row in tag_pred_vectors]
                support = sum(true_column)
                correct = sum(int(true_value == pred_value) for true_value, pred_value in zip(true_column, pred_column))
                tag_accuracy = correct / len(true_column) if true_column else 0.0
                tag_f1 = f1_score(true_column, pred_column, zero_division=0)

                tp = sum(1 for true_value, pred_value in zip(true_column, pred_column) if true_value == 1 and pred_value == 1)
                fp = sum(1 for true_value, pred_value in zip(true_column, pred_column) if true_value == 0 and pred_value == 1)
                fn = sum(1 for true_value, pred_value in zip(true_column, pred_column) if true_value == 1 and pred_value == 0)
                tn = sum(1 for true_value, pred_value in zip(true_column, pred_column) if true_value == 0 and pred_value == 0)

                tag_per_class.append({
                    "tag": tag_name,
                    "support": support,
                    "correct": correct,
                    "accuracy": tag_accuracy,
                    "f1": tag_f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                })

        evaluation = {
            "num_evaluated": len(evaluated_ids),
            "difficulty_correct_count": correct,
            "difficulty_incorrect_count": len(evaluated_ids) - correct,
            "difficulty_accuracy": difficulty_accuracy,
            "difficulty_error_count": len(evaluated_ids) - correct,
            "difficulty_macro_f1": difficulty_macro_f1,
            "difficulty_weighted_f1": difficulty_weighted_f1,
            "exact_tag_match_rate": (
                sum(int(row["missing_tags"] == [] and row["extra_tags"] == []) for row in comparison_rows)
                / len(comparison_rows)
            ),
            "exact_tag_match_count": sum(
                int(row["missing_tags"] == [] and row["extra_tags"] == []) for row in comparison_rows
            ),
            "avg_tag_precision": sum(row["precision"] for row in tag_metric_rows) / len(tag_metric_rows),
            "avg_tag_recall": sum(row["recall"] for row in tag_metric_rows) / len(tag_metric_rows),
            "avg_tag_f1": sum(row["f1"] for row in tag_metric_rows) / len(tag_metric_rows),
            "avg_tag_jaccard": sum(row["jaccard"] for row in tag_metric_rows) / len(tag_metric_rows),
            "tag_true_positive_count": total_tag_tp,
            "tag_false_positive_count": total_tag_fp,
            "tag_false_negative_count": total_tag_fn,
            "tag_micro_precision": tag_micro_precision,
            "tag_micro_recall": tag_micro_recall,
            "tag_micro_f1": tag_micro_f1,
            "tag_macro_precision": tag_macro_precision,
            "tag_macro_recall": tag_macro_recall,
            "tag_macro_f1": tag_macro_f1,
            "difficulty_per_class": difficulty_per_class,
            "tag_per_class": tag_per_class,
            "ground_truth_difficulty_distribution": {
                diff: true_difficulty_counter.get(diff, 0)
                for diff in DIFFICULTY_ORDER
                if true_difficulty_counter.get(diff, 0) > 0
            },
            "top_ground_truth_tags": dict(true_tag_counter.most_common(15)),
        }
        summary["evaluation"] = evaluation
        summary["headline_metrics"] = {
            "difficulty_accuracy": evaluation["difficulty_accuracy"],
            "difficulty_macro_f1": evaluation["difficulty_macro_f1"],
            "difficulty_weighted_f1": evaluation["difficulty_weighted_f1"],
            "tag_micro_f1": evaluation["tag_micro_f1"],
            "tag_macro_f1": evaluation["tag_macro_f1"],
            "num_evaluated": evaluation["num_evaluated"],
        }

    return (
        summary,
        difficulty_counter,
        true_difficulty_counter,
        tag_counter,
        true_tag_counter,
        confidence_values,
        tag_count_values,
        difficulty_true,
        difficulty_pred,
        comparison_rows,
        evaluation_tags,
    )


def generate_outputs(summary, difficulty_counter, true_difficulty_counter, tag_counter, true_tag_counter,
                     confidence_values, tag_count_values, difficulty_true, difficulty_pred,
                     comparison_rows, evaluation_tags, training_difficulty_counter, output_dir):
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "per_problem_comparison.jsonl", "w", encoding="utf-8") as f:
        for row in comparison_rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    difficulty_labels = [d for d in DIFFICULTY_ORDER if difficulty_counter.get(d, 0) > 0]
    difficulty_values = [difficulty_counter[d] for d in difficulty_labels]
    if difficulty_labels:
        save_bar_chart(
            difficulty_labels,
            difficulty_values,
            "Predicted Difficulty Distribution",
            "Problems",
            output_dir / "difficulty_distribution.png",
        )

    true_difficulty_labels = [d for d in DIFFICULTY_ORDER if true_difficulty_counter.get(d, 0) > 0]
    true_difficulty_values = [true_difficulty_counter[d] for d in true_difficulty_labels]
    if true_difficulty_labels:
        save_bar_chart(
            true_difficulty_labels,
            true_difficulty_values,
            "Ground Truth Difficulty Distribution",
            "Problems",
            output_dir / "ground_truth_difficulty_distribution.png",
        )

    training_difficulty_labels = [d for d in DIFFICULTY_ORDER if training_difficulty_counter.get(d, 0) > 0]
    training_difficulty_values = [training_difficulty_counter[d] for d in training_difficulty_labels]
    if training_difficulty_labels:
        save_bar_chart(
            training_difficulty_labels,
            training_difficulty_values,
            "Training Set Difficulty Distribution",
            "Problems",
            output_dir / "training_difficulty_distribution.png",
        )

    predicted_top_tags = tag_counter.most_common(20)
    if predicted_top_tags:
        save_bar_chart(
            [tag for tag, _ in predicted_top_tags],
            [count for _, count in predicted_top_tags],
            "Predicted Tag Distribution (Evaluated Set)",
            "Predicted occurrences",
            output_dir / "predicted_tag_distribution.png",
            rotate=True,
        )

    ground_truth_top_tags = true_tag_counter.most_common(20)
    if ground_truth_top_tags:
        save_bar_chart(
            [tag for tag, _ in ground_truth_top_tags],
            [count for _, count in ground_truth_top_tags],
            "Ground Truth Tag Distribution (Evaluated Set)",
            "True occurrences",
            output_dir / "ground_truth_tag_distribution.png",
            rotate=True,
        )

    if true_tag_counter:
        comparison_tags = [tag for tag, _ in true_tag_counter.most_common(20)]
        save_grouped_bar_chart(
            comparison_tags,
            [true_tag_counter.get(tag, 0) for tag in comparison_tags],
            [tag_counter.get(tag, 0) for tag in comparison_tags],
            "Ground truth",
            "Predicted",
            "Tag Distribution Comparison (Top 20 True Tags)",
            "Occurrences",
            output_dir / "tag_distribution_comparison_top20.png",
        )

    if difficulty_true and difficulty_pred:
        used_labels = [d for d in DIFFICULTY_ORDER if d in set(difficulty_true) | set(difficulty_pred)]
        save_confusion_matrix(
            difficulty_true,
            difficulty_pred,
            used_labels,
            output_dir / "difficulty_confusion_matrix.png",
        )

    evaluation = summary.get("evaluation", {})
    if evaluation.get("difficulty_per_class"):
        with open(output_dir / "difficulty_metrics_per_class.json", "w", encoding="utf-8") as f:
            json.dump(evaluation["difficulty_per_class"], f, indent=2)

    if evaluation.get("tag_per_class"):
        with open(output_dir / "tag_metrics_per_class.json", "w", encoding="utf-8") as f:
            json.dump(evaluation["tag_per_class"], f, indent=2)
        save_tag_confusion_heatmap(
            evaluation["tag_per_class"],
            output_dir / "tag_confusion_matrix_top20.png",
            top_k=20,
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze predictions.json and generate charts.")
    parser.add_argument(
        "--predictions",
        default="predictions.json",
        help="Path to predictions.json",
    )
    parser.add_argument(
        "--metadata",
        default="cleaned_data/dataset_splits.json",
        help="Optional ground truth metadata file",
    )
    parser.add_argument(
        "--output-dir",
        default="prediction_report",
        help="Directory for generated summary and graphics",
    )
    parser.add_argument(
        "--training-data",
        default="training_data/train.jsonl",
        help="Training split jsonl used for training-set distribution",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    training_data_path = Path(args.training_data)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions = load_json(predictions_path)
    if not predictions:
        print("predictions.json is empty. Generate predictions first, then rerun this script.")
        return

    ground_truth = load_ground_truth(metadata_path)
    training_difficulty_counter = load_training_difficulty_distribution(training_data_path)
    ensure_output_dir(output_dir)

    (
        summary,
        difficulty_counter,
        true_difficulty_counter,
        tag_counter,
        true_tag_counter,
        confidence_values,
        tag_count_values,
        difficulty_true,
        difficulty_pred,
        comparison_rows,
        evaluation_tags,
    ) = analyze_predictions(
        predictions,
        ground_truth,
    )
    generate_outputs(
        summary,
        difficulty_counter,
        true_difficulty_counter,
        tag_counter,
        true_tag_counter,
        confidence_values,
        tag_count_values,
        difficulty_true,
        difficulty_pred,
        comparison_rows,
        evaluation_tags,
        training_difficulty_counter,
        output_dir,
    )

    print(f"Saved report to {output_dir}")
    print(f"Summary JSON: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
