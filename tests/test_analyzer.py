import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import cv2
import pytest

# Import functions from your analyzer module
from universal_field_toolkit_analyzer_sred import (
    load_image,
    enhance_image,
    equalize_blue_channel,
    extract_brightness,
    extract_color_metrics,
    extract_texture,
    extract_edge_density,
    extract_shadow_intensity,
    classify_texture,
    update_csv,
    process_photo,
)

# ---------------------------------------------------------
# Helper: create synthetic test image
# ---------------------------------------------------------

def create_test_image(path, color=(100, 150, 200)):
    """Creates a simple 100x100 BGR image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = color
    cv2.imwrite(path, img)
    return img

# ---------------------------------------------------------
# Test load_image
# ---------------------------------------------------------

def test_load_image_reads_file_correctly(tmp_path):
    img_path = tmp_path / "test.jpg"
    create_test_image(str(img_path))
    img = load_image(str(img_path))
    assert img is not None
    assert img.shape == (100, 100, 3)

def test_load_image_missing_file_returns_none():
    img = load_image("nonexistent.jpg")
    assert img is None

# ---------------------------------------------------------
# Test enhance_image (updated)
# ---------------------------------------------------------

def test_enhance_image_changes_luminance():
    # Use a non-uniform image so histogram equalization actually changes values
    img = np.tile(np.arange(50, dtype=np.uint8), (50, 1))
    img = np.stack([img, img, img], axis=2)  # Make it 3-channel BGR

    enhanced = enhance_image(img)

    assert enhanced is not None
    assert enhanced.shape == img.shape
    assert not np.array_equal(img, enhanced)  # Now guaranteed to change

# ---------------------------------------------------------
# Test equalize_blue_channel (updated)
# ---------------------------------------------------------

def test_equalize_blue_channel_modifies_blue_only():
    # Use non-uniform blue channel so equalization changes it
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.arange(50, dtype=np.uint8), (50, 1))  # Blue gradient
    img[:, :, 1] = 100  # Green constant
    img[:, :, 2] = 150  # Red constant

    eq = equalize_blue_channel(img)

    assert eq.shape == img.shape
    assert not np.array_equal(eq[:, :, 0], img[:, :, 0])  # Blue changed
    assert np.array_equal(eq[:, :, 1], img[:, :, 1])      # Green unchanged
    assert np.array_equal(eq[:, :, 2], img[:, :, 2])      # Red unchanged

# ---------------------------------------------------------
# Test extract_brightness
# ---------------------------------------------------------

def test_extract_brightness_correct_value():
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    brightness = extract_brightness(img)
    assert brightness == pytest.approx(200.0)

# ---------------------------------------------------------
# Test extract_color_metrics
# ---------------------------------------------------------

def test_extract_color_metrics_returns_correct_means():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 0] = 10
    img[:, :, 1] = 20
    img[:, :, 2] = 30
    metrics = extract_color_metrics(img)
    assert metrics["mean_b"] == 10
    assert metrics["mean_g"] == 20
    assert metrics["mean_r"] == 30

# ---------------------------------------------------------
# Test extract_texture
# ---------------------------------------------------------

def test_extract_texture_nonzero():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[10:20, 10:20] = 255
    texture = extract_texture(img)
    assert texture > 0

# ---------------------------------------------------------
# Test extract_edge_density
# ---------------------------------------------------------

def test_extract_edge_density_detects_edges():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (40, 40), (255, 255, 255), 2)
    edges = extract_edge_density(img)
    assert edges > 0

# ---------------------------------------------------------
# Test extract_shadow_intensity
# ---------------------------------------------------------

def test_extract_shadow_intensity_dark_image():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    shadow = extract_shadow_intensity(img)
    assert shadow == 1.0

def test_extract_shadow_intensity_bright_image():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    shadow = extract_shadow_intensity(img)
    assert shadow == 0.0

# ---------------------------------------------------------
# Test classify_texture
# ---------------------------------------------------------

def test_classify_texture_grainy():
    assert classify_texture({"texture": 600}) == "grainy"

def test_classify_texture_smooth():
    assert classify_texture({"texture": 100}) == "smooth"

# ---------------------------------------------------------
# Test update_csv
# ---------------------------------------------------------

def test_update_csv_updates_correct_row(tmp_path, monkeypatch):
    csv_path = tmp_path / "field_data.csv"

    df = pd.DataFrame([{
        "Photo_Filename": "test_ingested.jpg",
        "Brightness": 0,
        "Mean_R": 0,
        "Mean_G": 0,
        "Mean_B": 0,
        "Texture": 0,
        "Edge_Density": 0,
        "Shadow_Intensity": 0,
        "Texture_Class": ""
    }])
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr("universal_field_toolkit_analyzer_sred.CSV_PATH", str(csv_path))

    features = {
        "brightness": 123,
        "mean_r": 10,
        "mean_g": 20,
        "mean_b": 30,
        "texture": 999,
        "edge_density": 50,
        "shadow_intensity": 0.5,
        "texture_class": "grainy"
    }

    update_csv("test_ingested.jpg", features)

    updated = pd.read_csv(csv_path)
    assert updated.loc[0, "Brightness"] == 123
    assert updated.loc[0, "Mean_R"] == 10
    assert updated.loc[0, "Texture_Class"] == "grainy"

def test_update_csv_missing_file(monkeypatch, capsys):
    monkeypatch.setattr("universal_field_toolkit_analyzer_sred.CSV_PATH", "/nonexistent.csv")
    update_csv("photo.jpg", {})
    captured = capsys.readouterr()
    assert "not found" in captured.out

def test_update_csv_missing_row(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "field_data.csv"
    pd.DataFrame([{"Photo_Filename": "other.jpg"}]).to_csv(csv_path, index=False)

    monkeypatch.setattr("universal_field_toolkit_analyzer_sred.CSV_PATH", str(csv_path))

    update_csv("missing.jpg", {})
    captured = capsys.readouterr()
    assert "No matching CSV row" in captured.out

# ---------------------------------------------------------
# Test process_photo
# ---------------------------------------------------------

def test_process_photo_creates_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("universal_field_toolkit_analyzer_sred.WORKING_DIR", str(tmp_path))
    monkeypatch.setattr("universal_field_toolkit_analyzer_sred.CSV_PATH", str(tmp_path / "field_data.csv"))

    df = pd.DataFrame([{"Photo_Filename": "sample_ingested.jpg"}])
    df.to_csv(tmp_path / "field_data.csv", index=False)

    img_path = tmp_path / "sample_ingested.jpg"
    create_test_image(str(img_path))

    process_photo(str(img_path))

    assert os.path.isfile(tmp_path / "sample_analyzed.jpg")
    assert os.path.isfile(tmp_path / "sample_enhanced.jpg")

    updated = pd.read_csv(tmp_path / "field_data.csv")
    assert "Brightness" in updated.columns
    assert updated.loc[0, "Brightness"] > 0



