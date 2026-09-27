# Project Context: Explainable AI (XAI) for Vision-Language Models (VLMs)

## 1. Project Overview
This project is for a Master's seminar in Explainable AI (NLP/Multimodal). The core research question is: **How do non-semantic linguistic perturbations (e.g., aggressive tone, emojis, camelCase, ALL CAPS) alter the visual grounding and self-attention mechanisms of unified Vision-Language Models?**

Instead of testing for performance metrics, this project uses XAI techniques to measure "attention drift"—quantifying how much a model's visual focus scatters when the text prompt contains stylistic noise, despite the semantic intent remaining identical.

## 2. Hardware Constraints (CRITICAL)
- **GPU:** NVIDIA RTX 3050 (Laptop/Desktop)
- **VRAM:** Strictly 6 GB
- **RAM:** 16 GB
- **Implication for Code:** We cannot load the model in standard FP16 because the base weights (~4.5 GB) plus the massive self-attention matrices will cause an Out of Memory (OOM) error. 
- **Solution:** The model MUST be loaded in **8-bit quantization** using `bitsandbytes` (`load_in_8bit=True`). The code must aggressively clear the CUDA cache (`torch.cuda.empty_cache()`) after every forward pass. 

## 3. Target Model & Architecture Details
- **Model:** `Qwen/Qwen2-VL-2B-Instruct`
- **Architectural Shift (Crucial):** Unlike older VLMs (e.g., BLIP-2), Qwen2-VL does NOT use a separate cross-attention mechanism. It dynamically compresses image patches into visual tokens and concatenates them with the text tokens into a **single sequence**. The model then processes everything using standard **decoder-only Self-Attention**.
- **Dynamic Resolution:** Qwen2-VL dynamically changes the number of image tokens based on resolution. To prevent OOM errors, the processor must be initialized with strict pixel limits (e.g., `min_pixels=256*28*28`, `max_pixels=512*28*28`).

## 4. Experimental Pipeline
The code needs to execute the following pipeline systematically:

### A. Data Setup
1. Load a base image (e.g., a picture of a red cup on a table).
2. Define a dictionary of prompt perturbations targeting the same object:
   - `baseline`: "Locate the red cup in this image."
   - `all_caps`: "LOCATE THE RED CUP IN THIS IMAGE."
   - `camel_case`: "loCaTe tHe rEd cUp iN tHiS iMaGe."
   - `aggressive`: "FIND THE F*CKING RED CUP NOW!!! 😡👇"

### B. VLM Inference & Attention Extraction
1. Use `AutoProcessor` and `Qwen2VLForConditionalGeneration`. Format inputs using the chat template.
2. Run a forward pass without gradient calculation (`torch.no_grad()`). Pass `output_attentions=True`.
3. **The XAI Logic:** - Locate the sequence index of the target text token (e.g., the word "cup").
   - Locate the sequence indices that correspond to the visual tokens (`<|vision_start|>` to `<|vision_end|>`).
   - Extract the self-attention weights from the final transformer layer. 
   - Slice this $N \times N$ matrix to isolate the 1D array of attention weights where the **target text token** attends backward to the **visual tokens**.

### C. Quantification (Mathematical Shift)
1. Normalize the extracted 1D attention arrays to create probability distributions.
2. Calculate the distance/divergence between the `baseline` visual attention map and the `perturbed` visual attention maps. 
3. Implement Kullback-Leibler (KL) Divergence to quantify the "drift":
   $$D_{KL}(P \parallel Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)$$
   (Where $P$ is the baseline attention array and $Q$ is the perturbed array, iterating over visual token indices $i$).

### D. Visualization
1. Map the 1D visual token array back into a 2D spatial grid (based on Qwen's patch size and spatial merging).
2. Upsample the 2D grid to the original image resolution.
3. Overlay the attention heatmap onto the original image using `matplotlib` and `seaborn`.
4. Output a side-by-side grid showing the original image alongside the heatmaps and the KL Divergence score for each prompt style.

## 5. Agent Instructions
When I ask you to generate code for this project, please:
- Prioritize VRAM efficiency. Use `bitsandbytes` 8-bit loading. Delete unused tensors (`del outputs, inputs`) and empty cache between iterations.
- Write modular code (separate functions for model loading, token indexing, attention slicing, and plotting).
- **Pay special attention** to how Qwen2-VL maps the 1D visual tokens back to 2D spatial coordinates for the heatmap. Add clear comments explaining the reshaping math.
- Ensure all dependencies (`torch`, `transformers`, `qwen-vl-utils`, `bitsandbytes`, `matplotlib`) are properly imported.