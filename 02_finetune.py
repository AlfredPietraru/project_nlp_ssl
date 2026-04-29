"""
Production-ready fine-tuning pipeline.
Optimized for $10-15 budget with RunPod A40 GPU.
"""
import json
import inspect
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType

# ============================================================================
# CONFIG
# ============================================================================
MODEL_NAME = "distilbert-base-uncased"  # 66M params, very fast
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10
MAX_LENGTH = 512
WARMUP_RATIO = 0.1

print(f"Config: {MODEL_NAME}, batch={BATCH_SIZE}, epochs={NUM_EPOCHS}")


def build_training_args(**kwargs):
    """Create TrainingArguments while tolerating version-specific keyword changes."""
    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters)
    normalized = dict(kwargs)

    if "evaluation_strategy" in normalized and "evaluation_strategy" not in supported and "eval_strategy" in supported:
        normalized["eval_strategy"] = normalized.pop("evaluation_strategy")

    filtered = {key: value for key, value in normalized.items() if key in supported}
    return TrainingArguments(**filtered)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
def prepare_data():
    """Load ground-truth-labeled prepared data from training_data/*.jsonl."""
    print("\n[1] Loading data...")
    
    data_dir = Path("/home/alf/nlp_ssl/training_data")
    all_tags = set()
    examples_dict = {}
    
    for split_name in ["train", "validation", "test"]:
        split_file = data_dir / f"{split_name}.jsonl"
        if not split_file.exists():
            print(f"  ⚠ Missing {split_file}")
            continue

        examples_dict[split_name] = []

        with open(split_file, encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)

                text = example.get("text", "").strip()
                difficulty = example.get("difficulty")
                tags = example.get("tags", [])

                if not text or not difficulty:
                    continue

                examples_dict[split_name].append({
                    "text": text,
                    "difficulty": difficulty,
                    "tags": tags,
                })

                for tag in tags:
                    all_tags.add(tag)
    
    all_tags = sorted(list(all_tags))
    
    # Create datasets
    datasets = {}
    for split_name, examples in examples_dict.items():
        if not examples:
            continue
        
        dataset = Dataset.from_dict({
            "text": [ex["text"] for ex in examples],
            "difficulty": [ex["difficulty"] for ex in examples],
            "tags": [ex["tags"] for ex in examples],
        })
        
        datasets[split_name] = dataset
        print(f"  {split_name:12}: {len(examples):4} examples")
    
    print(f"  Total tags: {len(all_tags)}")
    return datasets, all_tags

# ============================================================================
# STEP 2: TOKENIZE
# ============================================================================
def tokenize_data(datasets, all_tags, tokenizer):
    """Tokenize all examples"""
    print("\n[2] Tokenizing...")
    
    def preprocess_function(examples):
        encodings = tokenizer(
            examples["text"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
        )
        
        difficulty_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
        encodings["difficulty"] = [difficulty_map.get(d, 0) for d in examples["difficulty"]]
        
        # Binary encode tags
        tag_list = examples["tags"]
        tag_vectors = []
        for tags in tag_list:
            vec = [1.0 if tag in tags else 0.0 for tag in all_tags]
            tag_vectors.append(vec)
        
        encodings["tags"] = tag_vectors
        
        return encodings
    
    tokenized = {}
    for split_name, dataset in datasets.items():
        tokenized[split_name] = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=32,
            remove_columns=["text", "difficulty", "tags"]
        )
    
    return tokenized

# ============================================================================
# STEP 3: CUSTOM TRAINER FOR MULTI-TASK LOSS
# ============================================================================
class MultiTaskTrainer(Trainer):
    """Multi-task learning: difficulty + tags"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        tags = inputs.pop("tags", None)
        difficulty = inputs.pop("difficulty", None)

        if difficulty is None or tags is None:
            missing = []
            if difficulty is None:
                missing.append("difficulty")
            if tags is None:
                missing.append("tags")
            raise ValueError(f"Missing supervision fields in batch: {', '.join(missing)}")

        if not torch.is_tensor(difficulty):
            difficulty = torch.tensor(difficulty, dtype=torch.long, device=model.device)
        else:
            difficulty = difficulty.to(model.device).long()

        if not torch.is_tensor(tags):
            tags_tensor = torch.tensor(tags, dtype=torch.float32, device=model.device)
        else:
            tags_tensor = tags.to(model.device).float()
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # First 8 coords for difficulty, rest for tags
        difficulty_logits = logits[:, :8]
        tags_logits = logits[:, 8:]
        
        # Difficulty loss
        diff_loss = torch.nn.functional.cross_entropy(difficulty_logits, difficulty)
        
        # Tags loss
        tags_tensor = tags_tensor.to(logits.device)
        tags_loss = torch.nn.functional.binary_cross_entropy_with_logits(tags_logits, tags_tensor)
        
        loss = (diff_loss + tags_loss) / 2
        
        return (loss, outputs) if return_outputs else loss

# ============================================================================
# STEP 4: TRAIN
# ============================================================================
def train():
    """Main training loop"""
    print("\n" + "="*70)
    print("🚀 FINE-TUNING PIPELINE")
    print("="*70)
    
    # Load data
    datasets, all_tags = prepare_data()
    
    # Load model & tokenizer
    print("\n[3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    num_labels = 8 + len(all_tags)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Task labels: 8 (difficulty) + {len(all_tags)} (tags) = {num_labels}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        dtype=torch.float32,
    )
    
    # Apply LoRA
    print("  Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Tokenize
    tokenized = tokenize_data(datasets, all_tags, tokenizer)
    
    # Training args
    print("\n[4] Training configuration...")
    training_args = build_training_args(
        output_dir="/home/alf/nlp_ssl/model_output",
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=42,
        fp16=False,
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
        label_names=["difficulty", "tags"],
    )
    
    # Trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized.get("train"),
        eval_dataset=tokenized.get("validation"),
    )
    
    # Train!
    print("\n" + "="*70)
    print("⏱️  TRAINING IN PROGRESS")
    print("="*70)
    trainer.train()
    
    # Evaluate
    if "test" in tokenized:
        print("\n[5] Test evaluation...")
        results = trainer.evaluate(eval_dataset=tokenized["test"])
        print(f"  Test loss: {results['eval_loss']:.4f}")
    
    # Save
    print("\n[6] Saving model...")
    output_dir = Path("/home/alf/nlp_ssl/model_output/final_model")
    output_dir.mkdir(exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open(output_dir / "tags_mapping.json", "w", encoding="utf-8") as f:
        json.dump({
            "difficulties": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "tags": all_tags,
        }, f, indent=2)
    
    print(f"  ✓ Model saved to {output_dir}")
    
    print("\n" + "="*70)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"Next: python 03_inference.py")

if __name__ == "__main__":
    train()
