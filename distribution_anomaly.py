import os
import glob
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor


DEFECT_TYPES = ["good", "bent", "color", "flip", "scratch"]
MOODS = ["dark", "rusty", "greasy"] 

def extract_features(path, model, processor, device, limit=100):
    files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        files.extend(glob.glob(os.path.join(path, ext)))

    if not files: return None
    files = files[:limit]

    features = []
    for f in files:
        try:
            image = Image.open(f).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model.get_image_features(**inputs).squeeze().cpu().numpy()
                features.append(feat)
        except: continue

    return np.array(features) if features else None

def get_dist_data(root_path, defects, model, processor, device, center, is_generated=False, mood=""):
    dist_dict = {}
    for defect in defects:
        if is_generated:
            target_path = os.path.join(root_path, defect, mood)
        else:
            target_path = os.path.join(root_path, defect)

        feats = extract_features(target_path, model, processor, device)
        if feats is not None:
            # Cosine Distance 
            sim = cosine_similarity(feats, center)
            dists = 1 - sim.flatten()
            dist_dict[defect] = dists
    return dist_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_root', default="./metal_nut/test")
    parser.add_argument('--gen_root', default="./Result_MetalNut_Test")
    parser.add_argument('--output', default="class_distribution_comparison_fixed.png")
    args = parser.parse_args(args=[])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"

    print(f">>> Loading Model: {model_id}")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    #Use Original Good as Center (the normal original data)
    print("\n>>> Calculating Anchor...")
    path_good = os.path.join(args.orig_root, "good")
    feats_good = extract_features(path_good, model, processor, device, limit=200)
    if feats_good is None:
        print("Error: Original Good data missing.")
        return
    normal_center = np.mean(feats_good, axis=0, keepdims=True)

    scenarios = []
    all_values = [] 

    #Processing Original
    print(f"\n[1/4] Processing Original Set...")
    dists_orig = get_dist_data(args.orig_root, DEFECT_TYPES, model, processor, device, normal_center, is_generated=False)
    scenarios.append({"title": "Original Distribution", "data": dists_orig})
    for v in dists_orig.values(): all_values.extend(v)

    # Processing Generated Moods
    for i, mood in enumerate(MOODS):
        print(f"[{i+2}/4] Processing Generated Mood: {mood}...")
        dists_gen = get_dist_data(args.gen_root, DEFECT_TYPES, model, processor, device, normal_center, is_generated=True, mood=mood)
        scenarios.append({"title": f"Generated ({mood.upper()}) Distribution", "data": dists_gen})
        for v in dists_gen.values(): all_values.extend(v)


    global_min = min(all_values)
    global_max = max(all_values)
    margin = (global_max - global_min) * 0.1
    xlim_range = (max(0, global_min - margin), global_max + margin)
    print(f"\n>>> Global X-Axis Range: {xlim_range}")

    print("\n>>> Plotting Results...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = {"good": "black", "bent": "blue", "color": "green", "flip": "orange", "scratch": "red"}

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        data_map = scenario["data"]

        for defect, dists in data_map.items():
            fill = True if defect == "good" else False
            alpha = 0.2 if defect == "good" else 1.0
            ls = "-" if defect == "good" else "--"

            sns.kdeplot(dists, ax=ax, label=defect.upper(), color=colors.get(defect, "gray"),
                        fill=fill, alpha=alpha, linewidth=2, linestyle=ls, warn_singular=False)

        ax.set_title(scenario["title"], fontsize=14, fontweight='bold')
        ax.set_xlabel("Anomaly Score (Distance from Orig. Good)")
        ax.set_ylabel("Density")
        ax.set_xlim(xlim_range)

        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.show()
    print(f"\nPlot Saved: {args.output}")

if __name__ == "__main__":
    main()