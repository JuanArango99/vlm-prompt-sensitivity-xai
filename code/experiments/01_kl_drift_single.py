import gc
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from scipy.stats import entropy
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import get_vision_position_ids

# 1. Initialization
print("Loading model...")
quantization_config = BitsAndBytesConfig(load_in_8bit=True) # 8-bit as requested in context, 4-bit also works
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", device_map="auto", quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

for param in model.parameters():
    param.requires_grad = False

# 2. Hooking mechanism
# Context specifies gradient tracking. We will use block -10 as discovered to avoid attention sinks.
gradients_store = []
def backward_hook(module, grad_input, grad_output):
    gradients_store.append(grad_output[0].detach())

vision_tower = model.model.visual.blocks[-10]
vision_tower.register_full_backward_hook(backward_hook)

def get_saliency_map(image, prompt, target_word):
    """
    Computes a normalized probability distribution of visual token saliency 
    for a given prompt and target word.
    """
    global gradients_store
    gradients_store = []
    
    # Restrict resolution to prevent VRAM overflow
    image = image.copy()
    image.thumbnail((768, 768))

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    inputs["pixel_values"].requires_grad = True

    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits

    # Find the target word token to backpropagate from
    target_class = processor.tokenizer.encode(target_word)[0]
    next_token_logits = logits[0, -1, :]
    
    model.zero_grad()
    next_token_logits[target_class].backward()

    # Process gradients
    gradients = gradients_store[0] # Shape: (num_patches, hidden_dim)
    
    # Method from Context: Absolute value of gradients, mean across hidden dim
    saliency = torch.mean(torch.abs(gradients), dim=-1).squeeze().cpu().numpy()
    
    # Normalize to create a probability distribution (summing to 1) for KL Divergence
    saliency = saliency / (np.sum(saliency) + 1e-8)

    # Reshape into image grid using Qwen2-VL's 2x2 macro-block position IDs
    grid_thw = inputs["image_grid_thw"]
    pos_ids = get_vision_position_ids(grid_thw, 2, {})

    h, w = grid_thw[0][1].item(), grid_thw[0][2].item()
    saliency_img = np.zeros((h, w), dtype=np.float32)

    for i in range(len(saliency)):
        y, x = pos_ids[i].tolist()
        saliency_img[y, x] = saliency[i]

    # Clean up memory rigorously (6GB VRAM limit)
    del inputs, outputs, logits, next_token_logits, gradients
    gradients_store = []
    torch.cuda.empty_cache()
    gc.collect()

    return saliency_img, image

def compute_kl_divergence(baseline_map, perturbed_map):
    """Computes KL Divergence between two normalized probability maps."""
    # Flatten and add epsilon to avoid log(0)
    p = baseline_map.flatten() + 1e-12
    q = perturbed_map.flatten() + 1e-12
    return entropy(p, q)

def run_experiment():
    url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?ixlib=rb-4.0.3&w=400&q=80"
    original_image = Image.open(BytesIO(requests.get(url).content))
    
    target_word = "cat"

    print("Running baseline prompt...")
    baseline_prompt = "What animal is in this image?"
    baseline_map, resized_img = get_saliency_map(original_image, baseline_prompt, target_word)

    print("Running perturbed prompt...")
    perturbed_prompt = "wHaT aNiMaL iS iN tHiS iMaGe??? 😡"
    perturbed_map, _ = get_saliency_map(original_image, perturbed_prompt, target_word)

    # Compute KL Divergence
    kl_div = compute_kl_divergence(baseline_map, perturbed_map)
    print(f"KL Divergence between Baseline and Perturbed: {kl_div:.4f}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 99th percentile vmax to ignore outliers as per context
    vmax_base = np.percentile(baseline_map, 99)
    vmax_pert = np.percentile(perturbed_map, 99)

    # Resize maps to image dimensions for overlay
    base_resized = np.array(Image.fromarray(baseline_map).resize((resized_img.width, resized_img.height), Image.Resampling.BILINEAR))
    pert_resized = np.array(Image.fromarray(perturbed_map).resize((resized_img.width, resized_img.height), Image.Resampling.BILINEAR))

    axes[0].imshow(resized_img)
    axes[0].imshow(base_resized, cmap='jet', alpha=0.5, vmax=np.percentile(base_resized, 99))
    axes[0].set_title(f"Baseline\nPrompt: '{baseline_prompt}'")
    axes[0].axis('off')

    axes[1].imshow(resized_img)
    axes[1].imshow(pert_resized, cmap='jet', alpha=0.5, vmax=np.percentile(pert_resized, 99))
    axes[1].set_title(f"Perturbed (KL: {kl_div:.4f})\nPrompt: '{perturbed_prompt}'")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("../../data/results/experiments/saliency_drift_results.png")
    print("Saved experiment results to saliency_drift_results.png")

if __name__ == "__main__":
    run_experiment()
