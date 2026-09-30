import textwrap
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import get_vision_position_ids

print("Loading model...")
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", device_map="auto", quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

for param in model.parameters():
    param.requires_grad = False

activations_store = []
gradients_store = []

def forward_hook(module, input, output):
    activations_store.append(output.detach())

def backward_hook(module, grad_input, grad_output):
    gradients_store.append(grad_output[0].detach())

vision_tower = model.model.visual.blocks[-10]
vision_tower.register_forward_hook(forward_hook)
vision_tower.register_full_backward_hook(backward_hook)

def get_cam(image, prompt, target_word):
    global activations_store, gradients_store
    activations_store = []
    gradients_store = []
    
    img_copy = image.copy()
    img_copy.thumbnail((384, 384))

    messages = [
        {"role": "user", "content": [{"type": "image", "image": img_copy}, {"type": "text", "text": prompt}]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    inputs["pixel_values"].requires_grad = True

    outputs = model(**inputs)
    logits = outputs.logits

    target_class = processor.tokenizer.encode(target_word)[0]
    next_token_logits = logits[0, -1, :]
    
    model.zero_grad()
    next_token_logits[target_class].backward()

    activations = activations_store[0] 
    gradients = gradients_store[0] 
    
    weights = torch.mean(gradients, dim=0, keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = torch.relu(cam)
    
    saliency = cam.to(torch.float32).squeeze().cpu().numpy()
    if np.max(saliency) > 0:
        saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))

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

    return saliency_img, img_copy

def main():
    image = Image.open("../../data/images/cat.jpg")
    
    # Ablation testing: Nouns vs Emotion
    prompts = [
        "If you don't correctly identify the animal in this image, people will die. What is it?", # 1. Original (High Emotion + Noun)
        "What animal is in this image? There are people nearby",                                     # 2. Calm + Noun
        "If you don't correctly identify the animal, a catastrophic disaster will occur!",        # 3. High Emotion + NO Noun
        "What animal is in this image?"                                                           # 4. Baseline
    ]
    
    target_word = "animal"

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    for idx, prompt in enumerate(prompts):
        print(f"Testing Ablation: '{prompt}'")
        cam_map, resized_img = get_cam(image, prompt, target_word)
        
        cam_resized = np.array(Image.fromarray(cam_map).resize((resized_img.width, resized_img.height), Image.Resampling.BILINEAR))
        
        axes[idx].imshow(resized_img)
        axes[idx].imshow(cam_resized, cmap='jet', alpha=0.5)
        wrapped_title = "\n".join(textwrap.wrap(prompt, width=40))
        axes[idx].set_title(f"Prompt: '{wrapped_title}'", fontsize=18, pad=15)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig("../../data/results/experiments/ablation_results.png")
    print("Saved to ablation_results.png")

if __name__ == "__main__":
    main()
