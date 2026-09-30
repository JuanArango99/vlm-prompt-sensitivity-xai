# XAI Saliency Drift in Vision-Language Models

This repository contains the codebase and experimental results for a Master's seminar term paper in Explainable AI (XAI). The project investigates **"Gradient-Based Input Saliency for Vision-Language Models"**, specifically focusing on how non-semantic linguistic perturbations (e.g., tone, urgency, formatting) alter the physical visual grounding of the Qwen2-VL-2B model.

## Core Discoveries

Through a series of Grad-CAM and KL Divergence ablation experiments, this project identified three major phenomena in Vision-Language Models (VLMs):

1. **Image-Conditioned Prompt Sensitivity:** The robustness of a VLM's visual attention to non-semantic linguistic perturbations is not uniform across inputs. We observed vastly different magnitudes of saliency drift depending on the input image (e.g., high drift on one subject, near-zero on another). This indicates that visual grounding stability is heavily dependent on specific image features or the model's latent priors for that subject.
2. **Attention Sink Defaulting:** When a VLM is distracted by heavy, non-visual linguistic semantics (such as insults or extreme urgency), it loses visual confidence and dumps its attention into arbitrary "Attention Sinks" (register tokens) in the background.
3. **Panic-Induced Scattered Vision:** Extreme, existential semantic urgency (e.g., "people will die") completely destabilizes the model's cross-attention matrix. The language layers inject chaotic activation that forces the model to rapidly scan and highlight almost every patch in the image, destroying targeted visual grounding.

## Repository Structure

* **`code/experiments/`**: Contains the reproducible, sequentially numbered Python scripts for all experiments.
* **`data/images/`**: Source images for the experiments (`cat.jpg`, `komodo.jpg`).
* **`data/results/experiments/`**: The generated Grad-CAM heatmaps and visual comparisons.
* **`latex/`**: Source files and templates for the final term paper.
* **`deprecated/`**: Old sweeping scripts and tests used during early methodology discovery.

## Running the Experiments

The experiments are heavily optimized to run on consumer hardware (tested on an RTX 3050 6GB VRAM) by leveraging 4-bit quantization and aggressive VRAM garbage collection.

To run a clean batch of all 6 experiments:
```bash
cd code/experiments
./run_all.sh
```

## Setup Requirements

* `torch`
* `transformers`
* `bitsandbytes`
* `qwen_vl_utils`
* `scipy`
* `matplotlib`
* `Pillow`
