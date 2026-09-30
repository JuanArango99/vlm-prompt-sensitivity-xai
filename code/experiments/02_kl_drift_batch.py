import gc
import os
import glob
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
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", device_map="auto", quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

for param in model.parameters():
    param.requires_grad = False

# 2. Hooking mechanism
activations_store = []
gradients_store = []

def forward_hook(module, input, output):
    activations_store.append(output.detach())

def backward_hook(module, grad_input, grad_output):
    gradients_store.append(grad_output[0].detach())

vision_tower = model.model.visual.blocks[-10]
vision_tower.register_forward_hook(forward_hook)
vision_tower.register_full_backward_hook(backward_hook)

def get_saliency_map(image, prompt, target_word):
    global activations_store, gradients_store
    activations_store = []
    gradients_store = []
    
    # Restrict resolution to prevent VRAM overflow during backprop
    image = image.copy()
    image.thumbnail((384, 384))

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

    # Process gradients using Grad-CAM
    activations = activations_store[0] 
    gradients = gradients_store[0] 
    
    weights = torch.mean(gradients, dim=0, keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = torch.relu(cam)
    
    saliency = cam.to(torch.float32).squeeze().cpu().numpy()
    saliency = saliency / (np.sum(saliency) + 1e-8)

    grid_thw = inputs["image_grid_thw"]
    pos_ids = get_vision_position_ids(grid_thw, 2, {})

    h, w = grid_thw[0][1].item(), grid_thw[0][2].item()
    saliency_img = np.zeros((h, w), dtype=np.float32)

    for i in range(len(saliency)):
        y, x = pos_ids[i].tolist()
        saliency_img[y, x] = saliency[i]

    del inputs, outputs, logits, next_token_logits, activations, gradients
    activations_store = []
    gradients_store = []
    torch.cuda.empty_cache()
    gc.collect()

    return saliency_img, image

def compute_kl_divergence(baseline_map, perturbed_map):
    p = baseline_map.flatten() + 1e-12
    q = perturbed_map.flatten() + 1e-12
    return entropy(p, q)

def run_experiment():
    image_dir = "../../data/images"
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    target_word = "animal"
    
    if not image_paths:
        print("No images found in", image_dir)
        return

    print(f"Found {len(image_paths)} images to process...")
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"\nProcessing {img_name}...")
        original_image = Image.open(img_path)

        baseline_prompt = "What animal is in this image?"
        baseline_map, resized_img = get_saliency_map(original_image, baseline_prompt, target_word)

        perturbed_prompt = "wHaT aNiMaL iS iN tHiS iMaGe??? 😡"
        perturbed_map, _ = get_saliency_map(original_image, perturbed_prompt, target_word)

        kl_div = compute_kl_divergence(baseline_map, perturbed_map)
        print(f"KL Divergence for {img_name}: {kl_div:.4f}")

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
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
        out_name = f"../../data/results/experiments/result_{img_name.split('.')[0]}.png"
        plt.savefig(out_name)
        print(f"Saved {out_name}")

if __name__ == "__main__":
    run_experiment()
