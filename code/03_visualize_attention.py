import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gc
import os
import glob

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def get_target_token_indices(input_ids, processor, target_word):
    """Dynamically find the indices of tokens that make up the target word using character offsets."""
    full_text = ""
    token_offsets = []
    
    for tid in input_ids:
        s = processor.decode(tid).lower()
        start = len(full_text)
        full_text += s
        token_offsets.append((start, len(full_text)))
        
    target = target_word.lower()
    start_char = full_text.find(target)
    if start_char == -1:
        return []
    end_char = start_char + len(target)
    
    indices = []
    for i, (start, end) in enumerate(token_offsets):
        # If the token overlaps with the target word
        if start < end_char and end > start_char:
            indices.append(i)
            
    return indices

def generate_prompts(target_word):
    """Generate the baseline and perturbed prompts for a given target word."""
    def camel_case(s):
        return "".join([c.upper() if i % 2 == 1 else c.lower() for i, c in enumerate(s)])
        
    return {
        "baseline": f"Locate the {target_word.lower()}.",
        "all_caps": f"LOCATE THE {target_word.upper()}.",
        "camel_case": f"loCaTe tHe {camel_case(target_word)}.",
        "aggressive": f"FIND THE F*CKING {target_word.upper()} NOW!!! 😡"
    }

def process_image(model, processor, image_path, target_word, output_dir):
    print(f"\n===========================================")
    print(f"Processing image: {image_path} | Target: {target_word}")
    print(f"===========================================")
    
    orig_img = Image.open(image_path).convert("RGB")
    prompts = generate_prompts(target_word)
    img_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    
    results = {}
    baseline_attn = None
    
    for style, text_prompt in prompts.items():
        print(f"--- Style: {style} ---")
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
        
        target_idx_list = get_target_token_indices(inputs.input_ids[0], processor, target_word)
        if not target_idx_list:
            print(f"WARNING: Could not find target word '{target_word}' in the tokenized sequence for {style}.")
            continue
            
        print(f"Target word '{target_word}' found at token indices: {target_idx_list}")
        
        image_indices = torch.where(inputs.input_ids[0] == img_pad_id)[0]
        num_patches = len(image_indices)
        grid_size = int(np.sqrt(num_patches))
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions
            
            last_layer_attn = attentions[-1][0] # (heads, seq_len, seq_len)
            mean_heads_attn = last_layer_attn.mean(dim=0) # (seq_len, seq_len)
            
            # Average attention across the target tokens to all image tokens
            text_to_image_attn = mean_heads_attn[target_idx_list, :][:, image_indices].mean(dim=0).float().cpu()
            
            # Normalize to sum to 1
            prob_dist = text_to_image_attn / text_to_image_attn.sum()
            
            kl_div = 0.0
            if style == "baseline":
                baseline_attn = prob_dist
            else:
                eps = 1e-12
                kl_div = torch.sum(baseline_attn * torch.log(baseline_attn / (prob_dist + eps))).item()
                print(f"KL Divergence from baseline: {kl_div:.4f}")
            
            attn_grid = text_to_image_attn.view(grid_size, grid_size).numpy()
            
            results[style] = {
                "grid": attn_grid,
                "kl": kl_div
            }
                
        del outputs, inputs, attentions
        cleanup()

    print(f"\nPlotting results for {target_word}...")
    fig, axes = plt.subplots(1, len(prompts), figsize=(15, 4))
    if len(prompts) == 1:
        axes = [axes]
        
    for ax, (style, data) in zip(axes, results.items()):
        attn_grid = data["grid"]
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
    output_path = os.path.join(output_dir, f"{target_word}_attention_drift.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved heatmap visualization to {output_path}")

def main():
    cleanup()
    
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"Loading {model_id} for batch visualization...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    images_dir = os.path.join(data_dir, 'images')
    results_dir = os.path.join(data_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.jpeg"))
    
    for image_path in image_files:
        # Extract target word from filename (e.g., 'elephants.jpg' -> 'elephants')
        target_word = os.path.splitext(os.path.basename(image_path))[0]
        process_image(model, processor, image_path, target_word, results_dir)

if __name__ == "__main__":
    main()
