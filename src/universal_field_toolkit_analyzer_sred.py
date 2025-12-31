import os
import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Unified working directory (GitHub-safe)
# ---------------------------------------------------------
WORKING_DIR = os.environ.get("WORKING_DIR", "data/testing-input-output")
CSV_PATH = os.path.join(WORKING_DIR, "field_data.csv")
os.makedirs(WORKING_DIR, exist_ok=True)

# ---------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------
def load_image(path):
    return cv2.imread(path)

def enhance_image(img):
    """Histogram equalization on luminance channel (YUV space)."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def equalize_blue_channel(img):
    """Blue-channel specific equalization (Yukon refinement for ice signals)."""
    b, g, r = cv2.split(img)
    b_eq = cv2.equalizeHist(b)
    return cv2.merge((b_eq, g, r))

def extract_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def extract_color_metrics(img):
    b, g, r = cv2.split(img)
    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))
    # Relative color difference (invariant to global light cast)
    total = mean_r + mean_g + mean_b
    normalized_blue = mean_b / total if total > 0 else 0.0
    # Light source estimation proxy
    color_temp_proxy = 1000 * (mean_r / mean_b) if mean_b > 0 else 0.0
    light_source_flag = "artificial" if color_temp_proxy < 3000 or color_temp_proxy > 5000 else "natural"
    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "normalized_blue": normalized_blue,
        "color_temp_proxy": color_temp_proxy,
        "light_source_flag": light_source_flag
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

def compute_relative_metrics(img):
    """Quadrant-based relative brightness & texture to handle uneven lighting."""
    h, w = img.shape[:2]
    quadrants = [
        img[0:h//2, 0:w//2],
        img[0:h//2, w//2:w],
        img[h//2:h, 0:w//2],
        img[h//2:h, w//2:w]
    ]
    quad_brightness = [extract_brightness(q) for q in quadrants]
    global_bright = np.mean(quad_brightness)
    rel_bright = [b / global_bright if global_bright > 0 else 0.0 for b in quad_brightness]
    rel_texture = np.mean([extract_texture(q) for q in quadrants])
    return {
        "relative_brightness_variance": np.var(rel_bright),
        "relative_texture_variance": rel_texture / extract_texture(img) if extract_texture(img) > 0 else 0.0
    }

def extract_shadow_direction(img):
    """Sobel gradients → direction variance to detect artificial multi-source shadows."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    angles = np.arctan2(sobely, sobelx)
    hist, _ = np.histogram(angles, bins=8)
    variance = np.var(hist)
    shadow_variance_flag = "artificial" if variance > 1000 else "natural"  # Tunable threshold
    return {
        "shadow_direction_variance": variance,
        "shadow_variance_flag": shadow_variance_flag
    }

def classify_texture(features):
    """Simple 2-class for now — expandable later."""
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
    # Core features
    df.loc[idx, "Brightness"] = features["brightness"]
    df.loc[idx, "Mean_R"] = features["mean_r"]
    df.loc[idx, "Mean_G"] = features["mean_g"]
    df.loc[idx, "Mean_B"] = features["mean_b"]
    df.loc[idx, "Texture"] = features["texture"]
    df.loc[idx, "Edge_Density"] = features["edge_density"]
    df.loc[idx, "Shadow_Intensity"] = features["shadow_intensity"]
    df.loc[idx, "Texture_Class"] = features["texture_class"]
    # Lighting-invariant additions
    df.loc[idx, "Normalized_Blue"] = features["normalized_blue"]
    df.loc[idx, "Color_Temp_Proxy"] = features["color_temp_proxy"]
    df.loc[idx, "Light_Source_Flag"] = features["light_source_flag"]
    df.loc[idx, "Relative_Brightness_Variance"] = features["relative_brightness_variance"]
    df.loc[idx, "Relative_Texture_Variance"] = features["relative_texture_variance"]
    df.loc[idx, "Shadow_Direction_Variance"] = features["shadow_direction_variance"]
    df.loc[idx, "Shadow_Variance_Flag"] = features["shadow_variance_flag"]
    df.to_csv(CSV_PATH, index=False)
    print(f"✓ Updated CSV for {photo_filename}")

# ---------------------------------------------------------
# Main analyzer logic
# ---------------------------------------------------------
def process_photo(photo_path):
    base = os.path.basename(photo_path)
    name, ext = os.path.splitext(base)
    analyzed_name = f"{name.replace('_ingested', '')}_analyzed{ext}"
    enhanced_name = f"{name.replace('_ingested', '')}_enhanced{ext}"
    analyzed_path = os.path.join(WORKING_DIR, analyzed_name)
    enhanced_path = os.path.join(WORKING_DIR, enhanced_name)

    img = load_image(photo_path)
    if img is None:
        print(f"⚠️ Could not load {photo_path}")
        return

    enhanced = enhance_image(img)
    enhanced = equalize_blue_channel(enhanced)

    features = {}
    features["brightness"] = extract_brightness(enhanced)
    features.update(extract_color_metrics(enhanced))
    features["texture"] = extract_texture(enhanced)
    features["edge_density"] = extract_edge_density(enhanced)
    features["shadow_intensity"] = extract_shadow_intensity(enhanced)
    features["texture_class"] = classify_texture(features)

    # Lighting-invariant refinements
    features.update(compute_relative_metrics(enhanced))
    features.update(extract_shadow_direction(enhanced))

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
    processed = 0
    for filename in os.listdir(WORKING_DIR):
        if filename.lower().endswith(("_ingested.jpg", "_ingested.jpeg", "_ingested.png")):
            process_photo(os.path.join(WORKING_DIR, filename))
            processed += 1
    print(f"🎉 Analyzer complete — {processed} images processed. Outputs in {WORKING_DIR}")

if __name__ == "__main__":
    main()


