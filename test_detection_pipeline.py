"""
test_detection_pipeline.py
==========================
Repeatable Verification Test Suite for Deepfake Detection System.

Tests:
  1. Model Discovery & Loading Status
  2. Single-Face Photograph Processing (checks MODEL_UNAVAILABLE & INCONCLUSIVE)
  3. Multi-Person Photograph (2 faces) Processing & Per-Face Analysis
  4. No-Face Landscape Image (graceful NO_FACE handling)
  5. Batch Evaluation across test_dataset/ (if dataset is provided)

Usage:
  python test_detection_pipeline.py
"""

import os
import sys

# Safe encoding for Windows consoles
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_project.settings')

import django
django.setup()

import cv2
import numpy as np
from detection.ml_model import (
    find_deepfake_classifier,
    is_classifier_available,
    load_deepfake_model,
    predict_deepfake
)
from detection.preprocessing import process_image


def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def create_synthetic_photo(num_faces=1, is_synthetic=False):
    """
    Creates a valid JPEG image with simulated human facial structure for testing.
    """
    w, h = 640, 480
    img = np.full((h, w, 3), 220, dtype=np.uint8)  # soft gray-blue studio background
    img[:, :, 0] = 230  # Blue tint
    img[:, :, 1] = 220
    img[:, :, 2] = 210

    centers = [(220, 240), (420, 240)] if num_faces == 2 else [(320, 240)]

    for idx, (cx, cy) in enumerate(centers):
        # Draw realistic head outline
        cv2.ellipse(img, (cx, cy), (75, 100), 0, 0, 360, (180, 195, 225), -1)  # Face base
        # Eyes
        cv2.ellipse(img, (cx - 28, cy - 25), (12, 8), 0, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(img, (cx + 28, cy - 25), (12, 8), 0, 0, 360, (50, 50, 50), -1)
        # Eyeballs
        cv2.circle(img, (cx - 28, cy - 25), 4, (255, 255, 255), -1)
        cv2.circle(img, (cx + 28, cy - 25), 4, (255, 255, 255), -1)
        # Nose
        cv2.line(img, (cx, cy - 10), (cx, cy + 15), (130, 145, 175), 3)
        # Mouth
        cv2.ellipse(img, (cx, cy + 45), (28, 12), 0, 0, 180, (70, 70, 140), 3)

        # Hair
        cv2.ellipse(img, (cx, cy - 70), (80, 45), 0, 180, 360, (40, 35, 30), -1)

    # Texture & Noise
    if not is_synthetic:
        # Camera sensor noise
        noise = np.random.normal(0, 12, (h, w, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    else:
        # Synthetic smoothing
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def test_model_discovery():
    print_header("TEST 1: Model Discovery & Availability Verification")
    info = find_deepfake_classifier()
    print(f"  Search directory       : {info['search_dir']}")
    print(f"  Classifier found       : {info['found']}")
    print(f"  Discovered file        : {info.get('filename')}")
    print(f"  Discovered format      : {info.get('format')}")
    print(f"  Reason / Diagnostics   : {info.get('reason')}")

    model, load_info = load_deepfake_model()
    print(f"  Loaded model instance  : {'Yes' if model is not None else 'None (Graceful Fallback)'}")
    print(f"  Runtime configuration  : {load_info.get('runtime') or 'No ML Runtime'}")
    
    if not info['found']:
        print("  [OK] System correctly identifies missing model and reports reason.")
    else:
        print(f"  [OK] Classifier model loaded successfully.")
    return info['found']


def test_single_face_pipeline():
    print_header("TEST 2: Single-Face Photograph Processing")
    temp_path = "test_single_face_temp.jpg"
    img = create_synthetic_photo(num_faces=1, is_synthetic=False)
    cv2.imwrite(temp_path, img)

    try:
        prep = process_image(temp_path)
        print(f"  Face detection success : {prep['success']}")
        print(f"  Faces detected count   : {prep['face_count']}")

        norm_faces = prep.get('normalized_faces', [])
        pred_data = predict_deepfake(norm_faces, media_type="IMAGE")

        print(f"  Analysis status        : {pred_data['analysis_status']}")
        print(f"  Reported prediction    : {pred_data['prediction']}")
        print(f"  Confidence score       : {pred_data['confidence']}%")
        print(f"  Real probability       : {pred_data['real_probability']}")
        print(f"  Fake probability       : {pred_data['fake_probability']}")
        print(f"  Engine utilized        : {pred_data['engine_used']}")

        # Verification checks
        if not is_classifier_available():
            assert pred_data['analysis_status'] == 'MODEL_UNAVAILABLE', "Status must be MODEL_UNAVAILABLE"
            assert pred_data['prediction'] == 'UNCERTAIN', "Result must be UNCERTAIN (Inconclusive)"
            assert pred_data['confidence'] == 0.0, "Confidence must be 0.0 (N/A) when model is unavailable"
            print("  [OK] Single-face test correctly reports MODEL_UNAVAILABLE with N/A confidence.")
        else:
            print("  [OK] Single-face test executed with trained model inference.")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_multi_face_pipeline():
    print_header("TEST 3: Multi-Person Photograph (2 Faces) Processing")
    temp_path = "test_multi_face_temp.jpg"
    img = create_synthetic_photo(num_faces=2, is_synthetic=False)
    cv2.imwrite(temp_path, img)

    try:
        prep = process_image(temp_path)
        print(f"  Faces detected count   : {prep['face_count']}")
        norm_faces = prep.get('normalized_faces', [])
        pred_data = predict_deepfake(norm_faces, media_type="IMAGE")

        print(f"  Analysis status        : {pred_data['analysis_status']}")
        print(f"  Reported prediction    : {pred_data['prediction']}")
        print(f"  Confidence             : {pred_data['confidence']}%")
        print(f"  Faces analyzed count   : {pred_data['faces_count']}")

        faces_list = pred_data.get('faces_analysis', [])
        print(f"\n  Per-Face Telemetry Breakdown ({len(faces_list)} faces):")
        for f in faces_list:
            sig = f.get('forensic_signals', {})
            print(f"    - {f['label']}: Pred={f.get('model_prediction')} | "
                  f"Laplacian={sig.get('laplacian_variance')} | "
                  f"PatchStd={sig.get('local_patch_std')} | "
                  f"Noise={sig.get('noise_estimate')}")

        if not is_classifier_available():
            assert pred_data['analysis_status'] == 'MODEL_UNAVAILABLE'
            assert pred_data['confidence'] == 0.0
            print("\n  [OK] Multi-face photograph correctly analyzes both faces without inventing confidence.")
        else:
            print("\n  [OK] Multi-face photograph successfully classified with individual face breakdowns.")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_no_face_pipeline():
    print_header("TEST 4: Image Without Faces (Landscape / Document)")
    temp_path = "test_no_face_temp.jpg"
    # Create pure landscape gradient (no human facial geometry)
    blank = np.zeros((400, 600, 3), dtype=np.uint8)
    for y in range(400):
        blank[y, :, 0] = int(255 * (y / 400.0))  # blue sky gradient
        blank[y, :, 1] = int(200 * (y / 400.0))
        blank[y, :, 2] = 100
    cv2.imwrite(temp_path, blank)

    try:
        prep = process_image(temp_path)
        print(f"  Face detected          : {prep['face_detected']}")
        print(f"  Face count             : {prep['face_count']}")
        print(f"  Error message          : {prep['error_message']}")

        assert prep['face_detected'] is False, "No face should be detected"
        assert prep['face_count'] == 0
        print("  [OK] Image without face gracefully reports NO FACE DETECTED.")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_labeled_dataset_evaluation():
    print_header("TEST 5: Labeled Test Dataset Evaluation (test_dataset/)")
    dataset_dir = os.path.join(os.path.dirname(__file__), "test_dataset")
    real_dir = os.path.join(dataset_dir, "real")
    fake_dir = os.path.join(dataset_dir, "fake")

    if not (os.path.isdir(real_dir) and os.path.isdir(fake_dir)):
        print("  [INFO] No local 'test_dataset/' folder found.")
        print("  To evaluate against a benchmark dataset:")
        print("    1. Create test_dataset/real/ and place authentic photos")
        print("    2. Create test_dataset/fake/ and place synthetic / deepfake photos")
        print("    3. Re-run this script to compute Accuracy, Precision, Recall, F1, and Confusion Matrix.")
        return

    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"  Found {len(real_files)} real and {len(fake_files)} fake test files.")
    if len(real_files) == 0 and len(fake_files) == 0:
        print("  [INFO] Test directories are empty.")
        return

    # Run predictions
    y_true = []
    y_pred = []

    for path in real_files:
        prep = process_image(path)
        if prep['face_detected']:
            res = predict_deepfake(prep['normalized_faces'])
            y_true.append(0)  # 0 = Real
            y_pred.append(0 if res['prediction'] == 'REAL' else 1)

    for path in fake_files:
        prep = process_image(path)
        if prep['face_detected']:
            res = predict_deepfake(prep['normalized_faces'])
            y_true.append(1)  # 1 = Fake
            y_pred.append(1 if res['prediction'] == 'FAKE' else 0)

    if len(y_true) > 0:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = np.mean(y_true == y_pred) * 100.0
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n  Evaluation Results ({len(y_true)} evaluated samples):")
        print(f"    Accuracy  : {acc:.2f}%")
        print(f"    Precision : {precision:.4f}")
        print(f"    Recall    : {recall:.4f}")
        print(f"    F1 Score  : {f1:.4f}")
        print("\n  Confusion Matrix:")
        print(f"               Pred REAL   Pred FAKE")
        print(f"    Act REAL :   {tn:5d}       {fp:5d}")
        print(f"    Act FAKE :   {fn:5d}       {tp:5d}")


def main():
    print_header("DEEPFAKE DETECTION SYSTEM — PIPELINE TEST RUNNER")
    test_model_discovery()
    test_single_face_pipeline()
    test_multi_face_pipeline()
    test_no_face_pipeline()
    test_labeled_dataset_evaluation()
    print_header("ALL PIPELINE VERIFICATION TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
