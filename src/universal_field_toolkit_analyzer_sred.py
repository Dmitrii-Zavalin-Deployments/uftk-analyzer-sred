import os
import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Unified working directory (GitHub‑safe)
# ---------------------------------------------------------

WORKING_DIR = os.environ.get("WORKING_DIR", "data/testing-input-output")
CSV_PATH = os.path.join(WORKING_DIR, "field_data.csv")

# Ensure directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# ---------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------

def load_image(path):
    return cv2.imread(path)

def enhance_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def equalize_blue_channel(img):
    b, g, r = cv2.split(img)
    b_eq = cv2.equalizeHist(b)
    return cv2.merge((b_eq, g, r))

def extract_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def extract_color_metrics(img):
    b, g, r = cv2.split(img)
    return {
        "mean_r": float(np.mean(r)),
        "mean_g": float(np.mean(g)),
        "mean_b": float(np.mean(b)),
    }

def extract_texture(img):
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def extract_edge_density(img):
    edges = cv2.Canny(img, 100, 200)
    return float(np.sum(edges > 0))

def extract_shadow_intensity(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.sum(gray < 50)
    return float(dark_pixels) / gray.size

def classify_texture(features):
    return "grainy" if features["texture"] > 500 else "smooth"

# ---------------------------------------------------------
# CSV update logic
# ---------------------------------------------------------

def update_csv(photo_filename, features):
    if not os.path.isfile(CSV_PATH):
        print("⚠️ field_data.csv not found — analyzer cannot update metadata.")
        return

    df = pd.read_csv(CSV_PATH)

    row_index = df.index[df["Photo_Filename"] == photo_filename].tolist()
    if not row_index:
        print(f"⚠️ No matching CSV row for {photo_filename}")
        return

    idx = row_index[0]

    df.loc[idx, "Brightness"] = features["brightness"]
    df.loc[idx, "Mean_R"] = features["mean_r"]
    df.loc[idx, "Mean_G"] = features["mean_g"]
    df.loc[idx, "Mean_B"] = features["mean_b"]
    df.loc[idx, "Texture"] = features["texture"]
    df.loc[idx, "Edge_Density"] = features["edge_density"]
    df.loc[idx, "Shadow_Intensity"] = features["shadow_intensity"]
    df.loc[idx, "Texture_Class"] = features["texture_class"]

    df.to_csv(CSV_PATH, index=False)
    print(f"✓ Updated CSV for {photo_filename}")

# ---------------------------------------------------------
# Main analyzer logic
# ---------------------------------------------------------

def process_photo(photo_path):
    base = os.path.basename(photo_path)
    name, ext = os.path.splitext(base)

    analyzed_name = f"{name.replace('_ingested','')}_analyzed{ext}"
    enhanced_name = f"{name.replace('_ingested','')}_enhanced{ext}"

    analyzed_path = os.path.join(WORKING_DIR, analyzed_name)
    enhanced_path = os.path.join(WORKING_DIR, enhanced_name)

    img = load_image(photo_path)
    enhanced = enhance_image(img)
    enhanced = equalize_blue_channel(enhanced)

    features = {}
    features["brightness"] = extract_brightness(enhanced)
    features.update(extract_color_metrics(enhanced))
    features["texture"] = extract_texture(enhanced)
    features["edge_density"] = extract_edge_density(enhanced)
    features["shadow_intensity"] = extract_shadow_intensity(enhanced)
    features["texture_class"] = classify_texture(features)

    cv2.imwrite(analyzed_path, img)
    cv2.imwrite(enhanced_path, enhanced)

    update_csv(base, features)

    print(f"✓ Analyzed: {photo_path}")
    print(f"  → {analyzed_name}")
    print(f"  → {enhanced_name}")

def main():
    if not os.path.isdir(WORKING_DIR):
        print(f"❌ WORKING_DIR does not exist: {WORKING_DIR}")
        return

    for filename in os.listdir(WORKING_DIR):
        if filename.endswith("_ingested.jpg"):
            process_photo(os.path.join(WORKING_DIR, filename))

    print(f"🎉 Analyzer complete. Outputs written to {WORKING_DIR}")

if __name__ == "__main__":
    main()



