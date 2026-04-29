"""
Inference script: use the fine-tuned model to predict difficulty and tags.

By default, this script reads prepared examples from training_data/test.jsonl
and writes predictions to predictions.json.
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path("/home/alf/nlp_ssl/model_output/final_model")
DEFAULT_INPUT = Path("/home/alf/nlp_ssl/training_data/test.jsonl")
DEFAULT_OUTPUT = Path("/home/alf/nlp_ssl/predictions.json")
MAX_LENGTH = 512


def load_label_mapping():
    mapping_path = MODEL_DIR / "tags_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing tags mapping: {mapping_path}")

    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    difficulties = mapping.get("difficulties", ["A", "B", "C", "D", "E", "F", "G", "H"])
    tags = mapping.get("tags", [])
    difficulty_labels = {idx: label for idx, label in enumerate(difficulties)}
    return difficulty_labels, tags


def load_model(num_labels):
    """Load the base model and attach the saved LoRA adapter."""
    print("Loading model...")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_DIR}")

    adapter_config_path = MODEL_DIR / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {adapter_config_path}")

    with open(adapter_config_path, encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_text(text, model, tokenizer, device, difficulty_labels, all_tags, confidence_threshold=0.3):
    """Predict difficulty and tags from one prepared text example."""
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]

    num_difficulties = len(difficulty_labels)
    difficulty_logits = logits[:num_difficulties]
    tags_logits = logits[num_difficulties:]

    difficulty_probs = torch.softmax(difficulty_logits, dim=0)
    difficulty_idx = torch.argmax(difficulty_probs).item()
    difficulty = difficulty_labels.get(difficulty_idx, "Unknown")
    difficulty_conf = difficulty_probs[difficulty_idx].item()

    tags_probs = torch.sigmoid(tags_logits)
    predicted_tags = []
    tag_confidences = {}
    for tag, prob in zip(all_tags, tags_probs):
        prob_val = float(prob.item())
        if prob_val >= confidence_threshold:
            predicted_tags.append(tag)
            tag_confidences[tag] = prob_val

    return {
        "difficulty": difficulty,
        "difficulty_confidence": float(difficulty_conf),
        "tags": predicted_tags,
        "tag_confidences": tag_confidences,
        "all_difficulties": {
            difficulty_labels[i]: float(prob.item()) for i, prob in enumerate(difficulty_probs)
        },
        "all_tags_scores": {
            tag: float(prob.item()) for tag, prob in zip(all_tags, tags_probs)
        },
    }


def load_examples(input_file, limit=None):
    """Load prepared jsonl examples."""
    examples = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            if example.get("id") and example.get("text"):
                examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    return examples


def batch_predict(examples, model, tokenizer, device, difficulty_labels, all_tags, confidence_threshold=0.3):
    results = {}
    for example in examples:
        problem_id = example["id"]
        try:
            pred = predict_text(
                example["text"],
                model,
                tokenizer,
                device,
                difficulty_labels,
                all_tags,
                confidence_threshold=confidence_threshold,
            )
            results[problem_id] = pred
            print(f"✓ {problem_id}: {pred['difficulty']} | {', '.join(pred['tags'][:3])}")
        except Exception as exc:
            print(f"✗ {problem_id}: {exc}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference on prepared problem data.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input jsonl file")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output predictions json file")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of examples")
    parser.add_argument("--threshold", type=float, default=0.3, help="Tag confidence threshold")
    args = parser.parse_args()

    print("=" * 70)
    print("📊 INFERENCE: Predict Tags & Difficulty")
    print("=" * 70)

    input_file = Path(args.input)
    output_file = Path(args.output)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    difficulty_labels, all_tags = load_label_mapping()
    model, tokenizer, device = load_model(num_labels=len(difficulty_labels) + len(all_tags))
    examples = load_examples(input_file, limit=args.limit)

    print(f"\nRunning inference on {len(examples)} examples from {input_file}...")
    predictions = batch_predict(
        examples,
        model,
        tokenizer,
        device,
        difficulty_labels,
        all_tags,
        confidence_threshold=args.threshold,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print(f"\n✅ Predictions saved to {output_file}")
    print(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
