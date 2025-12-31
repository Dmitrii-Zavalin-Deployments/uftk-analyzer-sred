import os
import numpy as np
import pandas as pd
import cv2
import pytest

from universal_field_toolkit_analyzer_sred import (
    load_image,
    enhance_image,
    equalize_blue_channel,
    extract_brightness,
    extract_color_metrics,
    extract_texture,
    extract_edge_density,
    extract_shadow_intensity,
    compute_relative_metrics,
    extract_shadow_direction,
    classify_texture,
    update_csv,
    process_photo,
)

# ---------------------------------------------------------
# Helper: synthetic image generator
# ---------------------------------------------------------

def create_test_image(path, color=(100, 150, 200)):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = color
    cv2.imwrite(path, img)
    return img

# ---------------------------------------------------------
# load_image
# ---------------------------------------------------------

def test_load_image_reads_file_correctly(tmp_path):
    img_path = tmp_path / "test.jpg"
    create_test_image(str(img_path))
    img = load_image(str(img_path))
    assert img is not None
    assert img.shape == (100, 100, 3)

def test_load_image_missing_file_returns_none():
    assert load_image("missing.jpg") is None

# ---------------------------------------------------------
# enhance_image
# ---------------------------------------------------------

def test_enhance_image_changes_luminance():
    img = np.tile(np.arange(50, dtype=np.uint8), (50, 1))
    img = np.stack([img, img, img], axis=2)
    enhanced = enhance_image(img)
    assert enhanced is not None
    assert not np.array_equal(img, enhanced)

# ---------------------------------------------------------
# equalize_blue_channel
# ---------------------------------------------------------

def test_equalize_blue_channel_modifies_blue_only():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.arange(50, dtype=np.uint8), (50, 1))
    img[:, :, 1] = 100
    img[:, :, 2] = 150

    eq = equalize_blue_channel(img)

    assert not np.array_equal(eq[:, :, 0], img[:, :, 0])
    assert np.array_equal(eq[:, :, 1], img[:, :, 1])
    assert np.array_equal(eq[:, :, 2], img[:, :, 2])

# ---------------------------------------------------------
# extract_brightness
# ---------------------------------------------------------

def test_extract_brightness_correct_value():
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    assert extract_brightness(img) == pytest.approx(200.0)

# ---------------------------------------------------------
# extract_color_metrics
# ---------------------------------------------------------

def test_extract_color_metrics_values_and_flags():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 0] = 10
    img[:, :, 1] = 20
    img[:, :, 2] = 30

    m = extract_color_metrics(img)

    assert m["mean_b"] == 10
    assert m["mean_g"] == 20
    assert m["mean_r"] == 30
    assert m["normalized_blue"] == pytest.approx(10 / 60)
    assert m["color_temp_proxy"] == pytest.approx(1000 * (30 / 10))
    assert m["light_source_flag"] in ("natural", "artificial")

# ---------------------------------------------------------
# extract_texture
# ---------------------------------------------------------

def test_extract_texture_nonzero():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[10:20, 10:20] = 255
    assert extract_texture(img) > 0

# ---------------------------------------------------------
# extract_edge_density
# ---------------------------------------------------------

def test_extract_edge_density_detects_edges():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (40, 40), (255, 255, 255), 2)
    assert extract_edge_density(img) > 0

# ---------------------------------------------------------
# extract_shadow_intensity
# ---------------------------------------------------------

def test_extract_shadow_intensity_dark_image():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert extract_shadow_intensity(img) == 1.0

def test_extract_shadow_intensity_bright_image():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert extract_shadow_intensity(img) == 0.0

# ---------------------------------------------------------
# compute_relative_metrics
# ---------------------------------------------------------

def test_compute_relative_metrics_variances():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :50] = 200  # left bright, right dark

    rel = compute_relative_metrics(img)

    assert "relative_brightness_variance" in rel
    assert rel["relative_brightness_variance"] > 0
    assert "relative_texture_variance" in rel

# ---------------------------------------------------------
# extract_shadow_direction
# ---------------------------------------------------------

def test_extract_shadow_direction_variance():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.line(img, (0, 50), (99, 50), (255, 255, 255), 3)

    sd = extract_shadow_direction(img)

    assert "shadow_direction_variance" in sd
    assert "shadow_variance_flag" in sd
    assert sd["shadow_direction_variance"] >= 0

# ---------------------------------------------------------
# classify_texture
# ---------------------------------------------------------

def test_classify_texture_grainy():
    assert classify_texture({"texture": 600}) == "grainy"

def test_classify_texture_smooth():
    assert classify_texture({"texture": 100}) == "smooth"

# ---------------------------------------------------------
# update_csv
# ---------------------------------------------------------

def test_update_csv_updates_all_fields(tmp_path, monkeypatch):
    csv_path = tmp_path / "field_data.csv"

    df = pd.DataFrame([{"Photo_Filename": "test_ingested.jpg"}])
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
        "texture_class": "grainy",
        "normalized_blue": 0.3,
        "color_temp_proxy": 4000,
        "light_source_flag": "natural",
        "relative_brightness_variance": 0.1,
        "relative_texture_variance": 0.2,
        "shadow_direction_variance": 500,
        "shadow_variance_flag": "artificial",
    }

    update_csv("test_ingested.jpg", features)

    updated = pd.read_csv(csv_path)

    for key in [
        "Brightness", "Mean_R", "Mean_G", "Mean_B", "Texture",
        "Edge_Density", "Shadow_Intensity", "Texture_Class",
        "Normalized_Blue", "Color_Temp_Proxy", "Light_Source_Flag",
        "Relative_Brightness_Variance", "Relative_Texture_Variance",
        "Shadow_Direction_Variance", "Shadow_Variance_Flag"
    ]:
        assert key in updated.columns

def test_update_csv_missing_file(monkeypatch, capsys):
    monkeypatch.setattr("universal_field_toolkit_analyzer_sred.CSV_PATH", "/missing.csv")
    update_csv("photo.jpg", {})
    assert "not found" in capsys.readouterr().out

def test_update_csv_missing_row(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "field_data.csv"
    pd.DataFrame([{"Photo_Filename": "other.jpg"}]).to_csv(csv_path, index=False)
    monkeypatch.setattr("universal_field_toolkit_analyzer_sred.CSV_PATH", str(csv_path))
    update_csv("missing.jpg", {})
    assert "No matching CSV row" in capsys.readouterr().out

# ---------------------------------------------------------
# process_photo
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
    assert updated.loc[0, "Brightness"] > 0
    assert "Normalized_Blue" in updated.columns
    assert "Shadow_Direction_Variance" in updated.columns



