import os
import glob
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor

def extract_features(path, model, processor, device, label=""):
    files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        files.extend(glob.glob(os.path.join(path, ext)))

    if not files:
        return None

    features = []
    print(f"[Info] Extracting features for '{label}' ({len(files)} images)...")

    for f in tqdm(files, unit="img"):
        try:
            image = Image.open(f).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model.get_image_features(**inputs).squeeze().cpu().numpy()
                features.append(feat)
        except Exception:
            continue

    return np.array(features) if features else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_dir', type=str, required=True)
    parser.add_argument('--generated_root', type=str, required=True)
    parser.add_argument('--output', default="pca_result.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"

    print(f">>> Loading CLIP Model...")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    data_map = {}

    # Original Data
    if os.path.exists(args.original_dir):
        feats = extract_features(args.original_dir, model, processor, device, label="Original")
        if feats is not None:
            data_map["Original"] = feats

    # Generated Data
    if os.path.exists(args.generated_root):
        subfolders = sorted([f.path for f in os.scandir(args.generated_root) if f.is_dir()])
        for folder in subfolders:
            mood_name = os.path.basename(folder)
            feats = extract_features(folder, model, processor, device, label=mood_name)
            if feats is not None:
                data_map[mood_name] = feats

    # PCA & Plot
    if "Original" in data_map and len(data_map["Original"]) > 1:
        print("\n[Info] Fitting PCA...")
        pca = PCA(n_components=2)
        pca.fit(data_map["Original"])

        plt.figure(figsize=(10, 8))
        unique_labels = list(data_map.keys())
        colors = sns.color_palette("hls", len(unique_labels))

        for i, label in enumerate(unique_labels):
            feats = data_map[label]
            proj = pca.transform(feats)

            if label == "Original":
                c, m, a, z = 'black', 'o', 0.3, 0
            else:
                c, m, a, z = colors[i], 'x', 0.8, 10

            plt.scatter(proj[:, 0], proj[:, 1], c=[c], marker=m, alpha=a, label=label, s=40, zorder=z)

        plt.title("CLIP Feature Space (PCA)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(args.output)
        print(f"[Success] Saved to {args.output}")
    else:
        print("[Error] Original data missing or insufficient.")

if __name__ == "__main__":
    main()