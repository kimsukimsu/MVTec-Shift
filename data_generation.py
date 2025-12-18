import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# MVTec AD prompt
INDUSTRIAL_PROMPTS = {
    "dark": "make it look like it is in a dark industrial factory with shadows, cinematic lighting, low light",
    "greasy": "make the metal surface look greasy and wet, strong specular highlights, oil stains",
    "rusty": "add rusty texture on the metal surface, old machinery, weathering, oxidation",
    "noisy": "add industrial dust and film grain, dusty air, iso noise, fog"
}

# Generator class
class DataGenerator:
    def __init__(self, model_id="timbrooks/instruct-pix2pix", device="cuda"):
        print(f">>> Loading Model: {model_id}...")
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            safety_checker=None
        ).to(device)
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        self.device = device

    def generate(self, image, prompt):
        original_size = image.size
        process_resolution = (512, 512) 
        input_image = image.resize(process_resolution)
        
        result = self.pipeline(
            prompt, 
            image=input_image, 
            num_inference_steps=20, 
            
            image_guidance_scale=1.5, #this parameter indicate how much original images are preserved
            guidance_scale=7.5
        ).images[0]
        
        # return to original size 
        return result.resize(original_size)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic industrial moods for MVTec AD.")
    parser.add_argument("--source", type=str, required=True, help="Path to input images (e.g., mvtec_ad/metal_nut/train/good)")
    parser.add_argument("--output", type=str, default="./MVTec_Gen", help="Base directory to save generated images")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--max_images", type=int, default=None, help="Number of images to process (default: all)")
    parser.add_argument("--mood", type=str, default="all", choices=["dark", "greasy", "rusty", "noisy", "all"], help="Target environment mood")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    tasks = []
    if args.mood == "all":
        for m_type, prompt in INDUSTRIAL_PROMPTS.items():
            tasks.append({"type": m_type, "prompt": prompt})
    else:
        tasks.append({"type": args.mood, "prompt": INDUSTRIAL_PROMPTS[args.mood]})

    generator = DataGenerator(device=args.device)

    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' not found.")
        return

    all_files = sorted([f for f in os.listdir(args.source) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    target_files = all_files[:args.max_images] if args.max_images else all_files
    
    if len(target_files) == 0:
        print("Error: No images found in the source directory.")
        return

    print(f"\n>>> Found {len(all_files)} images. Processing {len(target_files)} images from '{args.source}'.\n")

    # image generation
    for task in tasks:
        mood_type = task['type']
        prompt = task['prompt']
        
        save_dir = os.path.join(args.output, f"{mood_type}")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[*] Task: {mood_type.upper()} | Prompt: '{prompt}'")
        
        for img_name in tqdm(target_files, desc=f"Generating {mood_type}", unit="img"):
            img_path = os.path.join(args.source, img_name)
            
            try:
                image = Image.open(img_path).convert('RGB')
                result_image = generator.generate(image, prompt)
            
                save_name = f"{os.path.splitext(img_name)[0]}_{mood_type}.png"
                result_image.save(os.path.join(save_dir, save_name))
                    
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        
        print(f"Saved to {save_dir}\n")

if __name__ == "__main__":
    main()