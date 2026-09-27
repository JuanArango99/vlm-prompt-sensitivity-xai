import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import gc

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    cleanup()
    
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"Loading {model_id} for attention extraction...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    image_path = "sample_image.jpg"
    
    prompts = {
        "baseline": "Locate the dog.",
        "all_caps": "LOCATE THE DOG.",
        "camel_case": "loCaTe tHe dOg.",
        "aggressive": "FIND THE F*CKING DOG NOW!!! 😡"
    }

    print("\nStarting perturbation pipeline...")
    
    for style, text_prompt in prompts.items():
        print(f"\n=== Processing style: {style} ===")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
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
        
        seq_len = inputs.input_ids.shape[1]
        print(f"Input sequence length: {seq_len}")
        
        # Run forward pass, requesting attentions
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
            attentions = outputs.attentions
            if attentions:
                last_layer_attn = attentions[-1] # Shape: (batch, heads, seq_len, seq_len)
                print(f"Extracted attention from {len(attentions)} layers.")
                print(f"Last layer attention shape: {last_layer_attn.shape}")
                
                # To isolate vision tokens, we look for vision token ID. 
                # Qwen2-VL usually replaces image with a sequence of vision tokens.
                # Let's count them by finding tokens not in the standard vocabulary, 
                # or specifically the vision token ID.
                # In Qwen2-VL, image_pad token is commonly used.
                try:
                    img_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                    num_img_tokens = (inputs.input_ids == img_pad_id).sum().item()
                    print(f"Number of `<|image_pad|>` tokens: {num_img_tokens}")
                except Exception as e:
                    print("Could not directly count image tokens via <|image_pad|>")

            else:
                print("No attentions returned.")
                
        # Critical: Clear VRAM for the next iteration
        del outputs
        del inputs
        if 'attentions' in locals():
            del attentions
        cleanup()

if __name__ == "__main__":
    main()
