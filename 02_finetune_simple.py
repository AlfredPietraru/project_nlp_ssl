"""
Complete fine-tuning pipeline for difficulty + tags prediction.
Simplified version using standard HuggingFace Trainer.
"""
import json
import inspect
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
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
# CONFIG - ADJUST BASED ON YOUR GPU
# ============================================================================
MODEL_NAME = "distilbert-base-uncased"  # Lightweight, 66M params (fastest)
# Alternatives: "microsoft/phi-2", "gpt2", "roberta-base"

BATCH_SIZE = 16  # Can increase if you have more VRAM
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10
MAX_LENGTH = 512  # Can reduce more for speed
WARMUP_RATIO = 0.1
USE_LORA = True  # Use LoRA for efficiency


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
# STEP 1: LOAD & PREPARE DATA
# ============================================================================
def prepare_data():
    """Load prepared train/validation/test jsonl files from training_data/."""
    
    print("\n" + "="*70)
    print("📂 LOADING DATA")
    print("="*70)
    
    data_dir = Path("/home/alf/nlp_ssl/training_data")
    all_tags = set()
    examples_dict = {}
    
    for split_name in ["train", "validation", "test"]:
        split_file = data_dir / f"{split_name}.jsonl"
        if not split_file.exists():
            print(f"⚠ Missing {split_file}")
            continue

        examples_dict[split_name] = []

        with open(split_file, encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)

                problem_id = example.get("id")
                text = example.get("text", "").strip()
                difficulty = example.get("difficulty")
                tags = example.get("tags", [])

                if not problem_id or not text or not difficulty:
                    continue

                examples_dict[split_name].append({
                    "text": text,
                    "difficulty": difficulty,
                    "tags": tags,
                    "id": problem_id,
                })

                for tag in tags:
                    all_tags.add(tag)
    
    # Create MultiLabelBinarizer
    sorted_tags = sorted(list(all_tags))
    mlb = MultiLabelBinarizer(classes=sorted_tags)
    mlb.fit([sorted_tags])  # Fit with all possible tags
    
    print(f"\nFound {len(sorted_tags)} unique tags")
    print(f"Top 15 tags: {sorted_tags[:15]}")
    
    # Convert to HuggingFace Datasets
    datasets = {}
    for split_name, examples in examples_dict.items():
        if not examples:
            continue
        
        # Convert tags to multi-hot vectors
        tags_binary = mlb.transform([ex["tags"] for ex in examples])
        
        dataset = Dataset.from_dict({
            "text": [ex["text"] for ex in examples],
            "difficulty": [ex["difficulty"] for ex in examples],
            "tags": tags_binary.tolist(),
            "tags_list": [ex["tags"] for ex in examples],
            "id": [ex["id"] for ex in examples]
        })
        
        datasets[split_name] = dataset
        print(f"✓ {split_name:12}: {len(examples):4} examples")
    
    print(f"\nTotal tags: {len(sorted_tags)}")
    print(f"Total examples: {sum(len(d) for d in datasets.values())}")
    
    return datasets, mlb, sorted_tags

# ============================================================================
# STEP 2: TOKENIZATION & PREPROCESSING
# ============================================================================
def tokenize_and_align(datasets, tokenizer, max_length=512):
    """Tokenize all examples"""
    
    print("\n" + "="*70)
    print("🔤 TOKENIZING")
    print("="*70)
    
    def preprocess_function(examples):
        encodings = tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        
        # Map difficulty to class index
        difficulty_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
        encodings["labels"] = [difficulty_map.get(d, 0) for d in examples["difficulty"]]
        
        # Store tags for custom loss calculation
        encodings["tags"] = examples["tags"]
        
        return encodings
    
    tokenized_datasets = {}
    for split_name, dataset in datasets.items():
        print(f"Tokenizing {split_name}...")
        tokenized_datasets[split_name] = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=32,
            remove_columns=["text", "difficulty", "tags_list", "id"]
        )
    
    return tokenized_datasets

# ============================================================================
# STEP 3: CUSTOM MODEL FOR MULTI-TASK LEARNING
# ============================================================================
class CustomTrainer(Trainer):
    """Custom trainer to handle multi-task loss (difficulty + tags)"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract tags before passing to model
        tags = inputs.pop("tags", None)
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Split logits: first 8 for difficulty, rest for tags
        num_difficulties = 8
        difficulty_logits = logits[:, :num_difficulties]
        tags_logits = logits[:, num_difficulties:] if logits.shape[1] > num_difficulties else None
        
        # Difficulty loss (cross-entropy)
        difficulty_loss = torch.nn.functional.cross_entropy(
            difficulty_logits,
            inputs["labels"]
        )
        
        # Tags loss (binary cross-entropy)
        tags_loss = 0
        if tags_logits is not None and tags is not None:
            tags_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                tags_logits,
                torch.tensor(tags, dtype=torch.float32, device=logits.device)
            )
        
        # Combined loss (equal weighting)
        loss = (difficulty_loss + tags_loss) / 2 if tags_loss else difficulty_loss
        
        if return_outputs:
            return loss, outputs
        return loss

# ============================================================================
# STEP 4: TRAINING
# ============================================================================
def train():
    """Main training function"""
    
    print("\n" + "="*70)
    print("🚀 STARTING FINE-TUNING")
    print("="*70)
    
    # Load and prepare data
    datasets, mlb, all_tags = prepare_data()
    
    # Load model
    print("\n" + "="*70)
    print("🤖 LOADING MODEL")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Num labels: {8 + len(all_tags)}")  # 8 difficulties + N tags
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=8 + len(all_tags),  # Multi-task: difficulties + tags
        torch_dtype=torch.float32,
    )
    
    # Apply LoRA
    if USE_LORA:
        print(f"\nApplying LoRA configuration...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"],  # DistilBERT attention projections
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Tokenize datasets
    tokenized_datasets = tokenize_and_align(datasets, tokenizer, MAX_LENGTH)
    
    # Training arguments
    print("\n" + "="*70)
    print("⚙️  TRAINING CONFIGURATION")
    print("="*70)
    
    training_args = build_training_args(
        output_dir="/home/alf/nlp_ssl/model_output",
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_dir="/home/alf/nlp_ssl/logs",
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        fp16=False,
        disable_tqdm=False,
        gradient_accumulation_steps=2,  # Simulate larger batch size
    )
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=tokenized_datasets.get("validation"),
    )

    # Train
    print("\n" + "=" * 70)
    print("⏱️  TRAINING IN PROGRESS")
    print("=" * 70)
    trainer.train()

    # Evaluate on test set
    if "test" in tokenized_datasets:
        print("\n" + "=" * 70)
        print("📊 TEST SET EVALUATION")
        print("=" * 70)
        test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        print(f"Test Loss: {test_results.get('eval_loss', 'N/A'):.4f}")

    # Save model
    print("\n" + "=" * 70)
    print("💾 SAVING MODEL & ARTIFACTS")
    print("=" * 70)

    output_dir = Path("/home/alf/nlp_ssl/model_output/final_model")
    output_dir.mkdir(exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save tag mapping for inference.
    with open(output_dir / "tags_mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tags": all_tags,
                "difficulties": ["A", "B", "C", "D", "E", "F", "G", "H"],
            },
            f,
            indent=2,
        )

    print(f"✓ Model saved to {output_dir}")
    print("✓ Tokenizer saved")
    print("✓ Tags mapping saved")

    print("\n" + "=" * 70)
    print("✅ FINE-TUNING COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run: python 03_inference.py")
    print("  2. Test on new problems")
    print("  3. Deploy to production")


if __name__ == "__main__":
    train()
