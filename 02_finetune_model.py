"""
Fine-tune a small LLM to predict difficulty and tags for programming problems.
Uses LoRA (Parameter-Efficient Fine-Tuning) to keep cost low.
"""
import json
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
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType

# ============================================================================
# CONFIG
# ============================================================================
MODEL_NAME = "microsoft/phi-2"  # Fast, efficient (~7B, but quantizable)
# Alternative: "distilbert-base-uncased"  # Even smaller (~66M)

MAX_LENGTH = 1024  # Truncate long texts
BATCH_SIZE = 4  # Small batch for limited VRAM
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WARMUP_STEPS = 100

# ============================================================================
# LOAD DATA
# ============================================================================
def load_training_data():
    """Load prepared jsonl files"""
    data_dir = Path("/home/alf/nlp_ssl/training_data")
    
    datasets = {}
    for split in ["train", "validation", "test"]:
        file_path = data_dir / f"{split}.jsonl"
        if not file_path.exists():
            print(f"⚠ {split} data not found")
            continue
        
        examples = []
        with open(file_path) as f:
            for line in f:
                examples.append(json.loads(line))
        
        datasets[split] = Dataset.from_dict({
            "text": [ex["text"] for ex in examples],
            "difficulty": [ex["difficulty"] for ex in examples],
            "tags": [ex["tags"] for ex in examples],
            "id": [ex["id"] for ex in examples]
        })
        print(f"✓ Loaded {split}: {len(examples)} examples")
    
    return datasets

# ============================================================================
# PREPARE DATA FOR TRAINING
# ============================================================================
def prepare_datasets(datasets, tokenizer, difficulty_labels, multi_label_binarizer):
    """
    Tokenize and prepare data for multi-task learning.
    Task 1: Difficulty classification (8 classes: A-H)
    Task 2: Tag prediction (multi-label)
    """
    
    def preprocess_function(examples):
        # Tokenize texts
        encodings = tokenizer(
            examples["text"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert difficulty to indices
        difficulty_ids = [difficulty_labels[d] for d in examples["difficulty"]]
        
        # Convert tags to multi-hot vectors
        tags_binary = multi_label_binarizer.transform(examples["tags"])
        
        encodings["difficulty_labels"] = difficulty_ids
        encodings["tags_labels"] = tags_binary.tolist()
        
        return encodings
    
    processed = {}
    for split, dataset in datasets.items():
        processed[split] = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=16,
            remove_columns=["text", "difficulty", "tags", "id"]
        )
    
    return processed

# ============================================================================
# CUSTOM MODEL CLASS
# ============================================================================
class MultiTaskLLMForTagAndDifficulty(torch.nn.Module):
    """Multi-task model: predict difficulty and tags simultaneously"""
    
    def __init__(self, base_model, num_difficulties=8, num_tags=30):
        super().__init__()
        self.base_model = base_model
        self.num_tags = num_tags
        
        hidden_size = base_model.config.hidden_size
        
        # Task 1: Difficulty (8-class classification)
        self.difficulty_head = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, num_difficulties)
        )
        
        # Task 2: Tags (multi-label sigmoid)
        self.tags_head = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_tags)
        )
        
        self.loss_fn_difficulty = torch.nn.CrossEntropyLoss()
        self.loss_fn_tags = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, input_ids, attention_mask, difficulty_labels=None, tags_labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Predict difficulty
        difficulty_logits = self.difficulty_head(cls_output)
        
        # Predict tags
        tags_logits = self.tags_head(cls_output)
        
        # Calculate losses if labels provided
        loss = None
        if difficulty_labels is not None and tags_labels is not None:
            difficulty_loss = self.loss_fn_difficulty(
                difficulty_logits,
                torch.tensor(difficulty_labels, device=cls_output.device)
            )
            tags_loss = self.loss_fn_tags(
                tags_logits,
                torch.tensor(tags_labels, dtype=torch.float32, device=cls_output.device)
            )
            loss = difficulty_loss + tags_loss  # Weighted sum
        
        return {
            "loss": loss,
            "difficulty_logits": difficulty_logits,
            "tags_logits": tags_logits
        }

# ============================================================================
# TRAINING
# ============================================================================
def train():
    """Fine-tune model with LoRA"""
    
    print("\n" + "="*70)
    print("🚀 STARTING FINE-TUNING PIPELINE")
    print("="*70)
    
    # 1. Load data
    print("\n[1/4] Loading data...")
    datasets = load_training_data()
    
    if not datasets:
        print("❌ No training data found. Run 01_prepare_data.py first!")
        return
    
    # 2. Load model and tokenizer
    print("\n[2/4] Loading base model...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max length: {MAX_LENGTH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=8,  # 8 difficulties: A-H
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # 3. Apply LoRA
    print("\n[3/4] Applying LoRA (Parameter-Efficient Fine-Tuning)...")
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Apply to attention layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. Training setup
    print("\n[4/4] Setting up training...")
    
    training_args = TrainingArguments(
        output_dir="/home/alf/nlp_ssl/model_output",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_dir='/home/alf/nlp_ssl/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
        seed=42,
        fp16=torch.cuda.is_available(),
        disable_tqdm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("validation"),
    )
    
    # Train!
    print("\n" + "="*70)
    print("⏱  TRAINING IN PROGRESS")
    print("="*70)
    trainer.train()
    
    # Save model
    print("\n" + "="*70)
    print("💾 SAVING MODEL")
    print("="*70)
    model.save_pretrained("/home/alf/nlp_ssl/model_output/final_model")
    tokenizer.save_pretrained("/home/alf/nlp_ssl/model_output/final_model")
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: /home/alf/nlp_ssl/model_output/final_model")
    
    # Test on test set
    if "test" in datasets:
        print("\n" + "="*70)
        print("📊 EVALUATING ON TEST SET")
        print("="*70)
        results = trainer.evaluate(eval_dataset=datasets["test"])
        print(f"  Test Loss: {results.get('eval_loss', 'N/A'):.4f}")

if __name__ == "__main__":
    train()
