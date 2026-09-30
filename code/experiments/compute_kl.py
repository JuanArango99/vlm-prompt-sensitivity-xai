import gc
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import entropy
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import get_vision_position_ids
import warnings
warnings.filterwarnings('ignore')

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", device_map="auto", quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
for param in model.parameters(): param.requires_grad = False

gradients_store = []
def backward_hook(module, grad_input, grad_output):
    gradients_store.append(grad_output[0].detach())
vision_tower = model.model.visual.blocks[-10]
vision_tower.register_full_backward_hook(backward_hook)

def get_saliency_map(image, prompt, target_word):
    global gradients_store
    gradients_store = []
    image = image.copy()
    image.thumbnail((384, 384))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    inputs["pixel_values"].requires_grad = True
    outputs = model(**inputs)
    logits = outputs.logits
    target_class = processor.tokenizer.encode(target_word)[0]
    next_token_logits = logits[0, -1, :]
    model.zero_grad()
    next_token_logits[target_class].backward()
    gradients = gradients_store[0]
    saliency = torch.mean(torch.abs(gradients), dim=-1).squeeze().cpu().to(torch.float32).numpy()
    saliency = saliency / (np.sum(saliency) + 1e-8)
    del inputs, outputs, logits, next_token_logits, gradients
    gradients_store = []
    torch.cuda.empty_cache()
    gc.collect()
    return saliency

def compute_kl_divergence(p, q):
    return entropy(p.flatten() + 1e-12, q.flatten() + 1e-12)

img = Image.open("../../data/images/cat.jpg")

# 1. Linguistic Style
print("--- Linguistic Style (Target: animal) ---")
b_map = get_saliency_map(img, "What animal is in this image?", "animal")
print("ALL CAPS:", compute_kl_divergence(b_map, get_saliency_map(img, "WHAT ANIMAL IS IN THIS IMAGE?", "animal")))
print("camelCase:", compute_kl_divergence(b_map, get_saliency_map(img, "wHaT aNiMaL iS iN tHiS iMaGe?", "animal")))
print("Emoji:", compute_kl_divergence(b_map, get_saliency_map(img, "What animal is in this image? 🐱❓", "animal")))

# 2. Emotion
print("--- Emotion / Urgency (Target: animal) ---")
print("PLEASE:", compute_kl_divergence(b_map, get_saliency_map(img, "Please tell me what animal is in this image, thank you.", "animal")))
print("QUICK:", compute_kl_divergence(b_map, get_saliency_map(img, "QUICK!! we only have 5 seconds!! what animal is in this image??", "animal")))
print("DISASTER:", compute_kl_divergence(b_map, get_saliency_map(img, "If you don't find the animal right now, a disaster will occur!", "animal")))

# 3. Ablation
print("--- Ablation (Target: disaster vs people) ---")
# Using the same setup as 06_noun_ablation
b_map2 = get_saliency_map(img, "Are there any people in this image?", "people")
t_map = get_saliency_map(img, "If you don't help me right now, a disaster will occur!", "disaster")
print("Calm People vs Urgent Disaster:", compute_kl_divergence(b_map2, t_map))

