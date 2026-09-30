#!/bin/bash
set -e

echo "Starting clean run of all experiments..."

echo "Running 01_kl_drift_single.py..."
python3 01_kl_drift_single.py

echo "Running 02_kl_drift_batch.py..."
python3 02_kl_drift_batch.py

echo "Running 03_semantic_attention.py..."
python3 03_semantic_attention.py

echo "Running 04_linguistic_style.py..."
python3 04_linguistic_style.py

echo "Running 05_emotion_threats.py..."
python3 05_emotion_threats.py

echo "Running 06_noun_ablation.py..."
python3 06_noun_ablation.py

echo "All experiments finished successfully!"
