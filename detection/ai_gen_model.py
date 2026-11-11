"""
AI-Generated Image Detection Module
====================================
Uses the onnx-community/ai-image-detect-distilled-ONNX model to classify
whether an image is a real photograph or AI-generated.

Model Details:
  - Repository: onnx-community/ai-image-detect-distilled-ONNX
  - Architecture: Distilled ViT (~11.8M parameters)
  - License: Apache-2.0
  - Training data: Midjourney, Stable Diffusion, DALL-E, and real photos
  - Task: Binary classification — Real vs AI-Generated (whole image, no face required)
  - Input: pixel_values [1, 3, 224, 224], standard ImageNet normalization
  - Output: logits [1, 2]
  - Class mapping: Must be verified from config.json at load time.
    Typical onnx-community convention: 0 = AI-generated / Fake, 1 = Real
    (Verified below via MODEL_CONFIG_PATH at runtime)

This module is intentionally separate from ml_model.py (the deepfake/face-swap
detector) because the two models serve different purposes:
  - ai_gen_model.py: Whole-image classifier for AI generation artifacts
  - ml_model.py: Face-crop classifier for face-swap / GAN manipulation artifacts

Never call this module for video face-swap detection.
"""

import os
import json
import numpy as np
import cv2

from django.conf import settings

# ─────────────────────────────────────────────────────────────────────────────
# Model file locations
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(settings.BASE_DIR, 'model')
AI_GEN_MODEL_DIR = os.path.join(MODEL_DIR, 'ai_gen_detector')
AI_GEN_MODEL_PATH = os.path.join(AI_GEN_MODEL_DIR, 'model.onnx')
AI_GEN_CONFIG_PATH = os.path.join(AI_GEN_MODEL_DIR, 'config.json')
AI_GEN_PREPROCESSOR_PATH = os.path.join(AI_GEN_MODEL_DIR, 'preprocessor_config.json')

# ─────────────────────────────────────────────────────────────────────────────
# Default class mapping (overridden at runtime from config.json if available)
# Convention for onnx-community/ai-image-detect-distilled-ONNX:
#   id2label: {"0": "artificial", "1": "nature"}
#   i.e. index 0 = AI-generated, index 1 = Real
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_AI_GEN_CLASS_MAPPING = {
    0: 'AI_GENERATED',
    1: 'REAL'
}

# Normalization parameters from preprocessor_config.json (mean=0.5, std=0.5)
VIT_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
VIT_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Module-level cache
_AI_GEN_SESSION = None
_AI_GEN_MODEL_INFO = None
_AI_GEN_CLASS_MAP = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_class_mapping():
    """
    Reads class labels from config.json in the model directory.
    Falls back to DEFAULT_AI_GEN_CLASS_MAPPING if the file is absent or malformed.
    Returns a dict {int_index: 'LABEL'}.
    """
    if os.path.isfile(AI_GEN_CONFIG_PATH):
        try:
            with open(AI_GEN_CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            id2label = cfg.get('id2label', {})
            if id2label:
                mapping = {}
                for k, v in id2label.items():
                    idx = int(k)
                    label_lower = v.strip().lower()
                    # Map common label strings to our internal constants
                    if label_lower in ('artificial', 'fake', 'ai', 'ai_generated', 'ai-generated'):
                        mapping[idx] = 'AI_GENERATED'
                    elif label_lower in ('nature', 'real', 'authentic', 'natural'):
                        mapping[idx] = 'REAL'
                    else:
                        # Unknown — keep original uppercased, treat as AI_GENERATED
                        mapping[idx] = label_lower.upper()
                print(f"[AI-Gen Model] Loaded class mapping from config.json: {mapping}")
                return mapping
        except Exception as e:
            print(f"[AI-Gen Model] Warning: Could not parse config.json — {e}. Using defaults.")
    return DEFAULT_AI_GEN_CLASS_MAPPING.copy()


def is_ai_gen_model_available():
    """Returns True if the AI-gen detector ONNX file exists on disk."""
    return os.path.isfile(AI_GEN_MODEL_PATH) and os.path.getsize(AI_GEN_MODEL_PATH) > 1024


def load_ai_gen_model():
    """
    Loads the AI-generated image detection ONNX model using ONNX Runtime.
    Returns (session, model_info_dict).
    Returns (None, model_info_dict) if the model file is not found or loading fails.
    Caches the session after first load.
    """
    global _AI_GEN_SESSION, _AI_GEN_MODEL_INFO, _AI_GEN_CLASS_MAP

    if _AI_GEN_SESSION is not None:
        return _AI_GEN_SESSION, _AI_GEN_MODEL_INFO

    if not is_ai_gen_model_available():
        _AI_GEN_MODEL_INFO = {
            'available': False,
            'error': (
                f"AI-gen detection model not found at {AI_GEN_MODEL_PATH}. "
                "Run: python manage.py shell -c \"from detection.ai_gen_model import download_ai_gen_model; download_ai_gen_model()\""
            ),
            'model_path': AI_GEN_MODEL_PATH,
            'class_mapping': DEFAULT_AI_GEN_CLASS_MAPPING,
        }
        return None, _AI_GEN_MODEL_INFO

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(AI_GEN_MODEL_PATH, providers=['CPUExecutionProvider'])

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        input_name = inputs[0].name if inputs else 'pixel_values'
        output_name = outputs[0].name if outputs else 'logits'
        input_shape = inputs[0].shape if inputs else [1, 3, 224, 224]
        output_shape = outputs[0].shape if outputs else [1, 2]

        class_mapping = _load_class_mapping()
        _AI_GEN_CLASS_MAP = class_mapping

        _AI_GEN_SESSION = session
        _AI_GEN_MODEL_INFO = {
            'available': True,
            'error': None,
            'runtime': f"ONNX Runtime {ort.__version__} (CPU)",
            'model_name': 'ai-image-detect-distilled-ONNX (onnx-community)',
            'model_path': AI_GEN_MODEL_PATH,
            'input_name': input_name,
            'output_name': output_name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'normalization': 'ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])',
            'class_mapping': class_mapping,
            'license': 'Apache-2.0',
        }
        print(f"[AI-Gen Model] Loaded: {AI_GEN_MODEL_PATH}")
        print(f"               Input: '{input_name}' {input_shape} | Output: '{output_name}' {output_shape}")
        print(f"               Class mapping: {class_mapping}")
        return _AI_GEN_SESSION, _AI_GEN_MODEL_INFO

    except Exception as e:
        err = f"Failed to load AI-gen ONNX model: {e}"
        print(f"[AI-Gen Model Error] {err}")
        _AI_GEN_MODEL_INFO = {
            'available': False,
            'error': err,
            'model_path': AI_GEN_MODEL_PATH,
            'class_mapping': DEFAULT_AI_GEN_CLASS_MAPPING,
        }
        return None, _AI_GEN_MODEL_INFO


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_ai_gen_detector(image_array):
    """
    Preprocesses an image array for the ViT-based AI-gen detector:
      1. Converts BGR (OpenCV default) to RGB.
      2. Resizes to 224x224 with bilinear interpolation.
      3. Rescales to [0.0, 1.0].
      4. Applies ImageNet normalization: (pixel - mean) / std.
      5. Transposes HWC -> CHW, adds batch dim -> [1, 3, 224, 224] float32.

    Args:
        image_array: numpy array, uint8 or float32, BGR or RGB, any spatial size.

    Returns:
        np.ndarray of shape (1, 3, 224, 224), dtype float32.
    """
    if image_array is None or image_array.size == 0:
        raise ValueError("Empty or None image passed to AI-gen preprocessor.")

    img = image_array.copy()

    # Normalise to uint8 range if float
    if img.dtype in (np.float32, np.float64) and img.max() <= 1.01:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # BGR -> RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = img[:, :, :3]

    # Resize
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Scale to [0, 1] and apply ViT normalization (mean=0.5, std=0.5)
    scaled = resized.astype(np.float32) / 255.0
    normalized = (scaled - VIT_MEAN) / VIT_STD

    # HWC -> CHW -> NCHW
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = np.expand_dims(chw, axis=0).astype(np.float32)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_ai_generated(image_path_or_array):
    """
    Runs the AI-generated image detector on a full image (not a face crop).

    Args:
        image_path_or_array: Either a file path (str) or a numpy BGR image array.

    Returns:
        dict with keys:
          - 'available'        : bool — whether the model ran successfully
          - 'prediction'       : 'AI_GENERATED' | 'REAL' | 'UNCERTAIN' | 'UNAVAILABLE'
          - 'ai_gen_probability': float in [0, 1] — probability image is AI-generated
          - 'real_probability' : float in [0, 1]
          - 'confidence'       : float in [0, 100] — model confidence percentage
          - 'engine_used'      : str
          - 'error'            : str or None
          - 'raw_logits'       : list[float] or None
    """
    session, model_info = load_ai_gen_model()

    if session is None or not model_info.get('available'):
        return {
            'available': False,
            'prediction': 'UNAVAILABLE',
            'ai_gen_probability': 0.0,
            'real_probability': 0.0,
            'confidence': 0.0,
            'engine_used': 'AI-Gen Detector (Unavailable)',
            'error': model_info.get('error', 'Model not loaded'),
            'raw_logits': None,
            'class_mapping': model_info.get('class_mapping', DEFAULT_AI_GEN_CLASS_MAPPING),
        }

    # Load image if path provided
    if isinstance(image_path_or_array, str):
        if not os.path.isfile(image_path_or_array):
            return {
                'available': False,
                'prediction': 'UNAVAILABLE',
                'ai_gen_probability': 0.0,
                'real_probability': 0.0,
                'confidence': 0.0,
                'engine_used': 'AI-Gen Detector',
                'error': f"Image file not found: {image_path_or_array}",
                'raw_logits': None,
                'class_mapping': model_info.get('class_mapping', DEFAULT_AI_GEN_CLASS_MAPPING),
            }
        image_array = cv2.imread(image_path_or_array)
        if image_array is None:
            return {
                'available': False,
                'prediction': 'UNAVAILABLE',
                'ai_gen_probability': 0.0,
                'real_probability': 0.0,
                'confidence': 0.0,
                'engine_used': 'AI-Gen Detector',
                'error': f"OpenCV could not read image: {image_path_or_array}",
                'raw_logits': None,
                'class_mapping': model_info.get('class_mapping', DEFAULT_AI_GEN_CLASS_MAPPING),
            }
    else:
        image_array = image_path_or_array

    try:
        tensor = preprocess_for_ai_gen_detector(image_array)
        input_name = model_info.get('input_name', 'pixel_values')
        output_name = model_info.get('output_name', 'logits')

        raw_outputs = session.run([output_name], {input_name: tensor})
        logits = np.squeeze(raw_outputs[0])

        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        class_mapping = _AI_GEN_CLASS_MAP or model_info.get('class_mapping', DEFAULT_AI_GEN_CLASS_MAPPING)

        # Identify which index corresponds to AI_GENERATED vs REAL
        ai_gen_idx = None
        real_idx = None
        for idx, label in class_mapping.items():
            if label == 'AI_GENERATED':
                ai_gen_idx = idx
            elif label == 'REAL':
                real_idx = idx

        # Graceful fallback if mapping doesn't align
        if ai_gen_idx is None:
            ai_gen_idx = 0
        if real_idx is None:
            real_idx = 1

        ai_gen_p = float(probs[ai_gen_idx]) if ai_gen_idx < len(probs) else 0.5
        real_p = float(probs[real_idx]) if real_idx < len(probs) else 0.5

        # Determine verdict with balanced threshold [0.46, 0.54]
        if 0.46 <= ai_gen_p <= 0.54:
            verdict = 'UNCERTAIN'
            confidence = round(abs(ai_gen_p - 0.5) * 200.0, 2)
        elif ai_gen_p > 0.54:
            verdict = 'AI_GENERATED'
            confidence = round(ai_gen_p * 100.0, 2)
        else:
            verdict = 'REAL'
            confidence = round(real_p * 100.0, 2)

        return {
            'available': True,
            'prediction': verdict,
            'ai_gen_probability': round(ai_gen_p, 4),
            'real_probability': round(real_p, 4),
            'confidence': confidence,
            'engine_used': f"onnx-community ai-image-detect-distilled-ONNX",
            'error': None,
            'raw_logits': logits.tolist() if hasattr(logits, 'tolist') else list(logits),
            'class_mapping': class_mapping,
        }

    except Exception as e:
        err = f"AI-gen inference error: {e}"
        print(f"[AI-Gen Model Error] {err}")
        return {
            'available': False,
            'prediction': 'UNAVAILABLE',
            'ai_gen_probability': 0.0,
            'real_probability': 0.0,
            'confidence': 0.0,
            'engine_used': 'AI-Gen Detector',
            'error': err,
            'raw_logits': None,
            'class_mapping': model_info.get('class_mapping', DEFAULT_AI_GEN_CLASS_MAPPING),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Download Helper (call from Django shell or management command)
# ─────────────────────────────────────────────────────────────────────────────

def download_ai_gen_model():
    """
    Downloads the AI-gen detection model from Hugging Face Hub into model/ai_gen_detector/.
    Call from Django shell:
        python manage.py shell -c "from detection.ai_gen_model import download_ai_gen_model; download_ai_gen_model()"

    Requires: pip install huggingface_hub
    License: Apache-2.0 (onnx-community/ai-image-detect-distilled-ONNX)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[Download] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    repo_id = 'onnx-community/ai-image-detect-distilled-ONNX'
    os.makedirs(AI_GEN_MODEL_DIR, exist_ok=True)

    files_to_download = [
        ('onnx/model.onnx', 'model.onnx'),
        ('config.json', 'config.json'),
        ('preprocessor_config.json', 'preprocessor_config.json'),
    ]

    success = True
    for repo_file, local_name in files_to_download:
        local_dest = os.path.join(AI_GEN_MODEL_DIR, local_name)
        if os.path.isfile(local_dest):
            print(f"[Download] Already exists: {local_dest}")
            continue
        print(f"[Download] Fetching {repo_file} from {repo_id} ...")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=repo_file,
                local_dir=AI_GEN_MODEL_DIR,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download may place file in a subfolder; move to flat structure
            if os.path.isfile(downloaded) and downloaded != local_dest:
                import shutil
                shutil.copy2(downloaded, local_dest)
                print(f"[Download] Saved to: {local_dest}")
            else:
                print(f"[Download] Saved to: {downloaded}")
        except Exception as e:
            print(f"[Download] Failed to download {repo_file}: {e}")
            success = False

    if success:
        print(f"\n[Download] Done. Model stored in: {AI_GEN_MODEL_DIR}")
        print("[Download] Run: python manage.py test detection  to verify integration.")
    else:
        print("[Download] Some files failed. Check network and retry.")

    return success
