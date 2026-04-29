#!/usr/bin/env python3
"""
Test script: Verify everything is set up correctly before running on GPU.
Run this locally first to catch any issues!
"""
import sys
from pathlib import Path

def test_data():
    """Check if data files exist"""
    print("\n[1] Checking data files...")
    base_path = Path("/home/alf/nlp_ssl/cleaned_data")
    
    if not base_path.exists():
        print(f"  ❌ Data path not found: {base_path}")
        return False
    
    splits_file = base_path / "dataset_splits.json"
    if not splits_file.exists():
        print(f"  ❌ dataset_splits.json not found")
        return False
    
    print(f"  ✓ Found {len(list(base_path.glob('*_*')))} problem folders")
    
    # Sample a few problems
    problems = list(base_path.glob('*_*'))[:5]
    for problem_path in problems:
        desc_file = problem_path / "description" / "description.txt"
        solutions_dir = problem_path / "solutions_c++"
        
        if desc_file.exists() and solutions_dir.exists():
            print(f"  ✓ {problem_path.name}: OK")
        else:
            print(f"  ⚠ {problem_path.name}: structure issue")
    
    return True

def test_python_env():
    """Check Python version and packages"""
    print("\n[2] Python environment...")
    print(f"  Python {sys.version.split()[0]}")
    
    required = ["torch", "transformers", "datasets", "peft", "sklearn"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} missing")
            missing.append(pkg)
    
    if missing:
        print(f"\n  ⚠ Install missing packages:")
        print(f"    pip install {' '.join(missing)}")
        return False
    
    return True

def test_gpu():
    """Check GPU availability"""
    print("\n[3] GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print(f"  ⚠ No GPU detected (will use CPU - slow)")
            print(f"  💡 For fast training, use RunPod A40")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    return True

def test_model_loading():
    """Quick model loading test"""
    print("\n[4] Model loading...")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print(f"  Loading DistilBERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print(f"  ✓ Tokenizer loaded")
        
        print(f"  Loading DistilBERT model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=30
        )
        print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
        
        # Test tokenization
        text = "Find the maximum element in an array"
        inputs = tokenizer(text, return_tensors="pt")
        print(f"  ✓ Tokenization works")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    return True

def test_data_loading():
    """Try loading and processing data"""
    print("\n[5] Data loading test...")
    try:
        import json
        
        base_path = Path("/home/alf/nlp_ssl/cleaned_data")
        splits_file = base_path / "dataset_splits.json"
        
        with open(splits_file) as f:
            data = json.load(f)
        
        # Check structure
        if "train" not in data or "validation" not in data or "test" not in data:
            print(f"  ❌ Invalid dataset_splits.json structure")
            return False
        
        train_count = len(data["train"])
        val_count = len(data["validation"])
        test_count = len(data["test"])
        
        print(f"  ✓ Train: {train_count} problems")
        print(f"  ✓ Val:   {val_count} problems")
        print(f"  ✓ Test:  {test_count} problems")
        
        # Try reading one problem
        sample = data["train"][0]
        problem_id = sample["id"]
        
        desc_file = base_path / problem_id / "description" / "description.txt"
        with open(desc_file, encoding='utf-8', errors='ignore') as f:
            desc_len = len(f.read())
        
        print(f"  ✓ Sample problem {problem_id}: {desc_len} chars")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    print("="*70)
    print("✅ PRE-FLIGHT CHECK")
    print("="*70)
    
    tests = [
        ("Data files", test_data),
        ("Python env", test_python_env),
        ("GPU", test_gpu),
        ("Model loading", test_model_loading),
        ("Data loading", test_data_loading),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    if all(r for _, r in results):
        print("\n✅ All checks passed! Ready to fine-tune.\n")
        print("Next steps:")
        print("  1. python 01_prepare_data.py    (CPU, ~2 min)")
        print("  2. python 02_finetune.py         (GPU, ~1 hour)")
        print("  3. python 03_inference.py        (GPU, ~1 min)")
        return 0
    else:
        print("\n❌ Some checks failed. See errors above.\n")
        print("Fixes:")
        print("  - Install packages: pip install -r requirements.txt")
        print("  - Check data path: /home/alf/nlp_ssl/cleaned_data")
        print("  - For GPU training: Use RunPod.io")
        return 1

if __name__ == "__main__":
    sys.exit(main())
