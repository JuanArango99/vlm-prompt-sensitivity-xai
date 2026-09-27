import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gc
import os

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def get_target_token_indices(input_ids, processor, style):
    """Find the indices of tokens that make up the target word 'dog', handling BPE splits."""
    # BPE tokenizes perturbations differently
    targets = {
        "baseline": [" dog"],
        "all_caps": [" do", "g"],
        "camel_case": [" d", "o", "g"],
        "aggressive": [" do", "g"]
    }
    
    target_tokens = targets[style]
    
    # Sliding window to find the sequence of tokens
    for i in range(len(input_ids) - len(target_tokens) + 1):
        match = True
        for j, t in enumerate(target_tokens):
            if processor.decode(input_ids[i+j]).lower() != t:
                match = False
                break
        if match:
            return list(range(i, i+len(target_tokens)))
            
    return []

def main():
    cleanup()
    
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"Loading {model_id} for visualization...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    image_path = "sample_image.jpg"
    orig_img = Image.open(image_path).convert("RGB")
    
    prompts = {
        "baseline": "Locate the dog.",
        "all_caps": "LOCATE THE DOG.",
        "camel_case": "loCaTe tHe dOg.",
        "aggressive": "FIND THE F*CKING DOG NOW!!! 😡"
    }

    img_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    
    results = {}
    baseline_attn = None

    print("\nStarting extraction and quantification pipeline...")
    
    for style, text_prompt in prompts.items():
        print(f"\n=== Processing style: {style} ===")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path,
                        "resized_height": 256,
                        "resized_width": 256
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        # Identify target tokens (the word "dog")
        target_idx_list = get_target_token_indices(inputs.input_ids[0], processor, style)
        if not target_idx_list:
            print(f"WARNING: Could not find target word 'dog' in the tokenized sequence for {style}.")
            continue
            
        print(f"Target word 'dog' found at token indices: {target_idx_list}")
        
        # Identify image tokens
        image_indices = torch.where(inputs.input_ids[0] == img_pad_id)[0]
        num_patches = len(image_indices)
        grid_size = int(np.sqrt(num_patches)) # Should be 9 for 81 patches
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            # Tuple of (batch, heads, seq_len, seq_len)
            attentions = outputs.attentions
            
            # Use the final layer
            last_layer_attn = attentions[-1][0] # (heads, seq_len, seq_len)
            
            # Average across all attention heads
            mean_heads_attn = last_layer_attn.mean(dim=0) # (seq_len, seq_len)
            
            # Extract attention from the target text tokens to all image tokens (average if multiple tokens)
            text_to_image_attn = mean_heads_attn[target_idx_list, :][:, image_indices].mean(dim=0).float().cpu()
            
            # Normalize to sum to 1 to represent a probability distribution for KL Divergence
            prob_dist = text_to_image_attn / text_to_image_attn.sum()
            
            # Calculate KL Divergence against baseline
            kl_div = 0.0
            if style == "baseline":
                baseline_attn = prob_dist
            else:
                # KL(P || Q) = sum(P * log(P / Q))
                # P = baseline, Q = perturbed
                # Add small epsilon to avoid log(0) or div by 0
                eps = 1e-12
                kl_div = torch.sum(baseline_attn * torch.log(baseline_attn / (prob_dist + eps))).item()
                print(f"KL Divergence from baseline: {kl_div:.4f}")
            
            # Reshape into 2D grid for visualization
            attn_grid = text_to_image_attn.view(grid_size, grid_size).numpy()
            
            results[style] = {
                "grid": attn_grid,
                "kl": kl_div
            }
                
        # Clean up memory
        del outputs
        del inputs
        del attentions
        cleanup()

    print("\nPlotting results...")
    # Plotting
    fig, axes = plt.subplots(1, len(prompts), figsize=(15, 4))
    if len(prompts) == 1:
        axes = [axes]
        
    for ax, (style, data) in zip(axes, results.items()):
        attn_grid = data["grid"]
        
        # Resize attention map to match original image using PIL
        attn_img = Image.fromarray(attn_grid).resize(orig_img.size, resample=Image.Resampling.BILINEAR)
        attn_resized = np.array(attn_img)
        
        ax.imshow(orig_img)
        ax.imshow(attn_resized, cmap='jet', alpha=0.5)
        
        title = f"{style}"
        if style != "baseline":
            title += f"\nKL: {data['kl']:.4f}"
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("attention_drift_results.png", dpi=300)
    print("Saved heatmap visualization to attention_drift_results.png")

if __name__ == "__main__":
    main()
