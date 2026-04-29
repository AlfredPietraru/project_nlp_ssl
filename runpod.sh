python3 -m venv venv
source venv/bin/activate
pip install uv
uv pip install torch transformers datasets peft scikit-learn pyyaml packaging accelerate tqdm matplotlib
python3 01_prepare_data.py
