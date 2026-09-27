import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import urllib.request
from PIL import Image
import os

def main():
    # Ensure memory is clean before loading
    torch.cuda.empty_cache()
    
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"Loading {model_id}...")
    
    # Load model in bfloat16 to fit in 6GB VRAM
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded successfully!")

    # Download a sample image if it doesn't exist
    image_path = "sample_image.jpg"
    if not os.path.exists(image_path):
        print("Downloading sample image...")
        image_url = "https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/demo.jpeg"
        urllib.request.urlretrieve(image_url, image_path)
    
    # Set up the conversation format required by Qwen2-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe what is in this image briefly."},
            ],
        }
    ]

    print("Processing inputs...")
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process image/video inputs
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Tokenize and process
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    print("Generating response...")
    # Generate output without computing gradients
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        
    # Trim the prompt from the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("\n--- Output ---")
    print(output_text[0])
    print("--------------")

if __name__ == "__main__":
    main()
