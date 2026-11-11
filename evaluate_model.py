"""
evaluate_model.py — Deepfake Classifier Evaluation & Verification Script
=========================================================================
Final-Year Diploma CSE Project — Model Verification Utility

Purpose:
  - Evaluates the trained SigLIP ONNX classifier on labeled test images.
  - Reports Accuracy, Precision, Recall, F1-Score, and per-class breakdown.
  - Generates a Confusion Matrix (saved as PNG).
  - Confirms class label mapping (Index 0 = FAKE, Index 1 = REAL).
  - Uses exact SigLIP normalization ((pixel/255.0 - 0.5) / 0.5).

Usage:
  python evaluate_model.py --test_dir test_dataset/
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from datetime import datetime

# Windows console safe unicode encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
EVAL_OUT_DIR = os.path.join(MODEL_DIR, 'evaluation')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
IMG_SIZE = 224


def find_model():
    """Find the classifier model file."""
    candidates = [
        ('deepfake_detector_v1.onnx', 'ONNX'),
        ('deepfake_model.onnx',        'ONNX'),
        ('deepfake_model.h5',          'H5'),
        ('deepfake_model.keras',       'KERAS'),
    ]
    for fname, fmt in candidates:
        p = os.path.join(MODEL_DIR, fname)
        if os.path.isfile(p) and os.path.getsize(p) > 1024:
            return p, fmt
    return None, None


def load_model(model_path, fmt):
    """Load model using ONNX Runtime or TensorFlow."""
    if fmt == 'ONNX':
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"[Eval] Loaded ONNX model via ONNX Runtime: {model_path}")
        print(f"       Input: '{input_name}', Output: '{output_name}'")
        return session, 'ONNX', input_name, output_name
    else:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            print(f"[Eval] Loaded Keras model (TF {tf.__version__}): {model_path}")
            return model, 'KERAS', 'input', 'output'
        except ImportError:
            print("[Eval ERROR] TensorFlow not installed. Cannot load .h5/.keras model.")
            return None, None, None, None


def preprocess_image_siglip(img_path):
    """
    Reads and standardizes an image using SigLIP preprocessing:
    - RGB color space
    - Resized to 224x224
    - Normalized: (x / 255.0 - 0.5) / 0.5 -> [-1.0, 1.0]
    - Shape: (1, 3, 224, 224), float32
    """
    try:
        with open(img_path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(chw, axis=0).astype(np.float32)
        return tensor
    except Exception as e:
        print(f"  [Warn] Could not load {img_path}: {e}")
        return None


def run_inference(model, fmt, tensor, input_name='pixel_values', output_name='logits'):
    """
    Runs model inference on preprocessed tensor.
    SigLIP Deepfake mapping:
      logits[0] -> FAKE (index 0)
      logits[1] -> REAL (index 1)
    Returns (fake_prob, real_prob).
    """
    if fmt == 'ONNX':
        outputs = model.run([output_name], {input_name: tensor})
        raw = np.squeeze(outputs[0])
    else:
        # Keras batch
        hwc = np.transpose(np.squeeze(tensor), (1, 2, 0))
        batch = np.expand_dims(hwc, axis=0)
        raw = np.squeeze(model.predict(batch, verbose=0))

    if raw.size == 2:
        exp_val = np.exp(raw - np.max(raw))
        probs = exp_val / np.sum(exp_val)
        fake_p = float(probs[0])
        real_p = float(probs[1])
    elif raw.size == 1:
        fake_p = float(raw)
        real_p = 1.0 - fake_p
    else:
        fake_p = 0.5
        real_p = 0.5

    return fake_p, real_p


def scan_test_dataset(test_dir):
    """Scans test_dir for 'real' and 'fake' subfolders."""
    items = []
    for label_str, is_fake_int in [('real', 0), ('fake', 1)]:
        folder = os.path.join(test_dir, label_str)
        if not os.path.isdir(folder):
            print(f"  [Warn] Subfolder not found: {folder}")
            continue
        for fname in os.listdir(folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                p = os.path.join(folder, fname)
                items.append((p, is_fake_int, label_str))
    return items


def compute_metrics(y_true, y_fake_probs, threshold=0.5):
    """Computes standard binary classification metrics."""
    tp, tn, fp, fn = 0, 0, 0, 0
    for yt, fp_val in zip(y_true, y_fake_probs):
        pred = 1 if fp_val >= threshold else 0
        if yt == 1 and pred == 1:
            tp += 1
        elif yt == 0 and pred == 0:
            tn += 1
        elif yt == 0 and pred == 1:
            fp += 1
        elif yt == 1 and pred == 0:
            fn += 1

    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'total': total,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'specificity': spec,
    }


def plot_confusion_matrix(tp, tn, fp, fn, save_path):
    """Plots and saves the confusion matrix image."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        matrix = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(5, 4.5))

        cax = ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.85)
        fig.colorbar(cax)

        classes = ['Real (0)', 'Fake (1)']
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)

        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if val > matrix.max() / 2 else 'black')

        plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
        plt.ylabel('Actual Ground Truth', fontsize=11, fontweight='bold')
        plt.title('Deepfake Classifier Confusion Matrix\n(SigLIP ONNX Model)', fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Eval] Confusion matrix image saved: {save_path}")
    except Exception as e:
        print(f"[Eval Warn] Could not plot confusion matrix: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection Classifier")
    parser.add_argument("--test_dir", type=str, default="test_dataset", help="Path to test directory containing real/ and fake/ subfolders")
    parser.add_argument("--model_path", type=str, default=None, help="Override model file path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for FAKE (default: 0.5)")
    args = parser.parse_args()

    print("=" * 70)
    print("DEEPFAKE CLASSIFIER - EVALUATION & VERIFICATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Directory: {os.path.abspath(args.test_dir)}")
    print(f"Threshold: >= {args.threshold:.2f} -> FAKE")

    # 1. Find model
    if args.model_path:
        model_path = args.model_path
        fmt = os.path.splitext(model_path)[1].upper().lstrip('.')
    else:
        model_path, fmt = find_model()

    if not model_path:
        print("\n[ERROR] No trained classifier found in model/ directory.")
        return

    print(f"\nModel: {model_path}  ({fmt})")

    # 2. Load model
    model, runtime, input_name, output_name = load_model(model_path, fmt)
    if model is None:
        return

    # 3. Scan test dataset
    if not os.path.isdir(args.test_dir):
        print(f"\n[ERROR] Test directory '{args.test_dir}' does not exist.")
        return

    items = scan_test_dataset(args.test_dir)
    if not items:
        print(f"[ERROR] No valid images found in {args.test_dir}/real/ or {args.test_dir}/fake/")
        return

    real_count = sum(1 for _, yt, _ in items if yt == 0)
    fake_count = sum(1 for _, yt, _ in items if yt == 1)
    print(f"\n[Eval] Found {len(items)} labeled test images.")
    print(f"  Real: {real_count}  |  Fake: {fake_count}")

    # 4. Run inference
    print(f"\n[Eval] Running inference on {len(items)} images...")
    y_true = []
    y_fake_probs = []
    errors = 0

    for img_path, true_int, label_str in items:
        tensor = preprocess_image_siglip(img_path)
        if tensor is None:
            errors += 1
            continue
        try:
            fake_p, real_p = run_inference(model, runtime, tensor, input_name, output_name)
            y_true.append(true_int)
            y_fake_probs.append(fake_p)
        except Exception as e:
            print(f"  [Error] Inference failed for {os.path.basename(img_path)}: {e}")
            errors += 1

    if not y_true:
        print("[ERROR] No images were successfully evaluated.")
        return

    print(f"  Successfully evaluated: {len(y_true)} images  |  Skipped/Error: {errors}")

    # 5. Compute metrics
    metrics = compute_metrics(y_true, y_fake_probs, args.threshold)

    # 6. Check for label reversal
    reversed_accuracy = 1.0 - metrics['accuracy']
    label_reversed = reversed_accuracy > metrics['accuracy']

    # 7. Print report
    print("\n" + "-" * 70)
    print("CLASSIFICATION METRICS (FAKE = positive class)")
    print("-" * 70)
    print(f"  Accuracy        : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision       : {metrics['precision'] * 100:.2f}%  (of FAKE predictions, how many are correct)")
    print(f"  Recall          : {metrics['recall'] * 100:.2f}%  (of actual FAKEs, how many detected)")
    print(f"  F1-Score        : {metrics['f1_score'] * 100:.2f}%")
    print(f"  Specificity     : {metrics['specificity'] * 100:.2f}%  (correct REAL classifications)")
    print("-" * 70)
    print(f"  Confusion Matrix:")
    print(f"    True Positives  (FAKE correctly detected)  : {metrics['tp']}")
    print(f"    True Negatives  (REAL correctly classified): {metrics['tn']}")
    print(f"    False Positives (REAL misclassified as FAKE): {metrics['fp']}")
    print(f"    False Negatives (FAKE misclassified as REAL): {metrics['fn']}")
    print("-" * 70)

    if label_reversed:
        print("\n[WARN]  ACCURACY IS BELOW 50% - POSSIBLE LABEL REVERSAL DETECTED!")
        print(f"   If labels were reversed, accuracy would be {reversed_accuracy * 100:.2f}%.")
        print("   Verify: output index 0 = FAKE, output index 1 = REAL.")
    elif metrics['accuracy'] > 0.75:
        print(f"\n[OK]  Model correctly oriented. Accuracy: {metrics['accuracy'] * 100:.2f}%")
    else:
        print(f"\n[NOTE]  Measured benchmark accuracy ({metrics['accuracy'] * 100:.2f}%).")

    # 8. Save outputs
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)
    cm_path = os.path.join(EVAL_OUT_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn'], cm_path)

    report_path = os.path.join(EVAL_OUT_DIR, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DEEPFAKE CLASSIFIER EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test images: {len(y_true)} evaluated, {errors} errors\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        f.write(f"Accuracy:    {metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"Precision:   {metrics['precision'] * 100:.2f}%\n")
        f.write(f"Recall:      {metrics['recall'] * 100:.2f}%\n")
        f.write(f"F1-Score:    {metrics['f1_score'] * 100:.2f}%\n")
        f.write(f"Specificity: {metrics['specificity'] * 100:.2f}%\n\n")
        f.write(f"TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}\n")
    print(f"[Eval] Full report saved: {report_path}")

    # Save class mapping
    class_map = {
        "model_repository": "pranjal-pravesh/prithivML-deepfake-detector-model-v1",
        "architecture": "SiglipForImageClassification",
        "input_tensor_name": input_name,
        "output_tensor_name": output_name,
        "class_labels": {
            "0": "FAKE",
            "1": "REAL"
        },
        "preprocessing": "SigLIP Normalization: (RGB / 255.0 - 0.5) / 0.5 -> [-1.0, 1.0]",
        "threshold": args.threshold
    }
    class_map_path = os.path.join(EVAL_OUT_DIR, 'class_mapping.json')
    with open(class_map_path, 'w', encoding='utf-8') as f:
        json.dump(class_map, f, indent=2)
    print(f"[Eval] Class mapping saved: {class_map_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
