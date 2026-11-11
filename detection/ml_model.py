import os
import cv2
import numpy as np
from django.conf import settings

# ─────────────────────────────────────────────────────────────────────────────
# Model Directory & Discovery Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(settings.BASE_DIR, 'model')
KNOWN_DETECTOR_FILES = {'face_detection_yunet_2023mar.onnx', 'deepfake_model.onnx'}  # deepfake_model.onnx is a dummy — skip it

# Class Mapping confirmed from Hugging Face model card & config.json:
# Model: pranjal-pravesh/prithivML-deepfake-detector-model-v1 (SigLIP backbone)
# id2label: {0: "Fake", 1: "Real"}
CLASS_MAPPING = {
    0: 'FAKE',
    1: 'REAL'
}

_LOADED_SESSION = None
_LOADED_MODEL_INFO = None


def find_deepfake_classifier():
    """
    Scans the model/ directory for trained classifier files:
      1. deepfake_detector_v1.onnx (Genuine Hugging Face SigLIP model)
      2. deepfake_model.onnx
      3. deepfake_model.h5 / .keras
    """
    if not os.path.isdir(MODEL_DIR):
        return {
            'found': False,
            'path': None,
            'filename': None,
            'format': None,
            'size_bytes': 0,
            'search_dir': MODEL_DIR,
            'reason': f"Model directory does not exist: {MODEL_DIR}"
        }

    priority_names = [
        'deepfake_detector_v1.onnx',
        'deepfake_model.h5',
        'deepfake_model.keras'
    ]

    for name in priority_names:
        p = os.path.join(MODEL_DIR, name)
        if os.path.isfile(p) and os.path.getsize(p) > 1024:
            ext = os.path.splitext(name)[1].lower().replace('.', '').upper()
            return {
                'found': True,
                'path': p,
                'filename': name,
                'format': ext,
                'size_bytes': os.path.getsize(p),
                'search_dir': MODEL_DIR,
                'reason': f"Found classifier model: {name}"
            }

    # Secondary scan for any other .onnx, .h5, or .keras file
    for fname in os.listdir(MODEL_DIR):
        if fname.lower() in KNOWN_DETECTOR_FILES:
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in ('.onnx', '.h5', '.keras'):
            p = os.path.join(MODEL_DIR, fname)
            if os.path.isfile(p) and os.path.getsize(p) > 1024:
                return {
                    'found': True,
                    'path': p,
                    'filename': fname,
                    'format': ext.replace('.', '').upper(),
                    'size_bytes': os.path.getsize(p),
                    'search_dir': MODEL_DIR,
                    'reason': f"Found alternative model file: {fname}"
                }

    return {
        'found': False,
        'path': None,
        'filename': None,
        'format': None,
        'size_bytes': 0,
        'search_dir': MODEL_DIR,
        'reason': "No classifier model (.onnx / .h5 / .keras) found in model/ directory."
    }


def is_classifier_available():
    """Returns True if a valid classifier model exists on disk."""
    info = find_deepfake_classifier()
    return info['found']


def load_deepfake_model():
    """
    Safely loads the trained classifier using ONNX Runtime for .onnx models
    or TensorFlow for legacy .h5/.keras models.
    Dynamically verifies input and output tensor names.
    """
    global _LOADED_SESSION, _LOADED_MODEL_INFO
    if _LOADED_SESSION is not None:
        return _LOADED_SESSION, _LOADED_MODEL_INFO

    discovery = find_deepfake_classifier()
    if not discovery['found']:
        _LOADED_MODEL_INFO = {
            'available': False,
            'error': discovery['reason'],
            'runtime': None,
            'input_name': None,
            'output_name': None,
            'input_shape': None,
            'output_shape': None,
            'path': None,
            'filename': None,
        }
        return None, _LOADED_MODEL_INFO

    model_path = discovery['path']
    fmt = discovery['format']

    # Case A: ONNX Model via ONNX Runtime
    if fmt == 'ONNX':
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            input_name = inputs[0].name if inputs else 'pixel_values'
            output_name = outputs[0].name if outputs else 'logits'
            input_shape = inputs[0].shape if inputs else (1, 3, 224, 224)
            output_shape = outputs[0].shape if outputs else (1, 2)

            _LOADED_SESSION = session
            _LOADED_MODEL_INFO = {
                'available': True,
                'error': None,
                'runtime': f"ONNX Runtime {ort.__version__} (CPU)",
                'format': 'ONNX',
                'model_name': 'prithivML-deepfake-detector-v1 (SigLIP)',
                'input_name': input_name,
                'output_name': output_name,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'path': model_path,
                'filename': discovery['filename'],
                'normalization': 'SigLIP ((pixel/255.0 - 0.5) / 0.5)',
            }
            print(f"[AI Model] Successfully loaded ONNX model: {discovery['filename']}")
            print(f"           Input: '{input_name}' {input_shape} | Output: '{output_name}' {output_shape}")
            return _LOADED_SESSION, _LOADED_MODEL_INFO

        except Exception as e:
            err = f"Failed to load ONNX model via ONNX Runtime: {e}"
            print(f"[AI Model Error] {err}")
            _LOADED_MODEL_INFO = {
                'available': False,
                'error': err,
                'runtime': 'ONNX Runtime',
                'path': model_path,
                'filename': discovery['filename'],
            }
            return None, _LOADED_MODEL_INFO

    # Case B: Legacy Keras / TensorFlow (.h5, .keras)
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        _LOADED_SESSION = model
        _LOADED_MODEL_INFO = {
            'available': True,
            'error': None,
            'runtime': f"TensorFlow {tf.__version__} (Keras)",
            'format': fmt,
            'model_name': 'Keras CNN Classifier',
            'input_name': 'input',
            'output_name': 'output',
            'input_shape': getattr(model, 'input_shape', (None, 224, 224, 3)),
            'output_shape': getattr(model, 'output_shape', None),
            'path': model_path,
            'filename': discovery['filename'],
            'normalization': 'Standard [0, 1]',
        }
        return _LOADED_SESSION, _LOADED_MODEL_INFO
    except Exception as e:
        err = f"TensorFlow model could not be loaded: {e}"
        _LOADED_MODEL_INFO = {
            'available': False,
            'error': err,
            'runtime': None,
            'path': model_path,
            'filename': discovery['filename'],
        }
        return None, _LOADED_MODEL_INFO


# ─────────────────────────────────────────────────────────────────────────────
# SigLIP Preprocessing Implementation
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_siglip(image_or_crop):
    """
    Standard SigLIP Image Preprocessing:
    1. Ensures RGB color space (converts from BGR if needed).
    2. Resizes to (224, 224) with bilinear interpolation.
    3. Rescales pixel values to [0, 1].
    4. Normalizes with mean=0.5, std=0.5:
       pixel_norm = (pixel / 255.0 - 0.5) / 0.5 -> mapped to range [-1.0, 1.0].
    5. Transposes from HWC (224, 224, 3) to CHW (3, 224, 224).
    6. Adds batch dimension: (1, 3, 224, 224), float32.
    """
    if image_or_crop is None or image_or_crop.size == 0:
        raise ValueError("Invalid image crop passed to preprocessor.")

    crop = image_or_crop.copy()
    
    # If image is float in [0, 1], bring back to uint8 [0, 255] for standard color conversion
    if crop.dtype in (np.float32, np.float64) and crop.max() <= 1.01:
        crop = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
    else:
        crop = crop.astype(np.uint8)

    # Convert to RGB (OpenCV default is BGR)
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    elif len(crop.shape) == 2:
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    else:
        rgb = crop[:, :, :3]

    # Resize to exact 224x224 expected by SigLIP
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Rescale & SigLIP Normalization
    rescaled = resized.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    normalized = (rescaled - mean) / std

    # HWC -> CHW -> NCHW
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = np.expand_dims(chw, axis=0).astype(np.float32)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary Forensic Metric Calculations
# ─────────────────────────────────────────────────────────────────────────────

def _compute_local_patch_std(gray, patch_size=16):
    """Measures variance of local texture patches (micro-texture smoothness)."""
    h, w = gray.shape
    stds = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = gray[y:y + patch_size, x:x + patch_size]
            stds.append(float(np.std(patch)))
    return float(np.mean(stds)) if stds else 0.0


def _compute_noise_estimate(gray):
    """Estimates sensor-level high-frequency residual noise."""
    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = gray.astype(np.float32) - blur
    return float(np.std(residual))


def _compute_ela_score(crop_uint8):
    """Lightweight Error Level Analysis proxy."""
    try:
        import io
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cv2.cvtColor(crop_uint8, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=75)
        buf.seek(0)
        reloaded = np.array(PILImage.open(buf))
        reloaded_bgr = cv2.cvtColor(reloaded, cv2.COLOR_RGB2BGR)
        diff = np.abs(crop_uint8.astype(np.float32) - reloaded_bgr.astype(np.float32))
        return float(np.mean(diff))
    except Exception:
        return 0.0


def calculate_forensic_metrics_for_crop(crop):
    """
    Computes computer-vision forensic measurements for a single face crop.
    Separated from the model prediction to provide transparent auxiliary indicators.
    """
    if crop.dtype in (np.float32, np.float64) and crop.max() <= 1.01:
        crop_uint8 = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
    else:
        crop_uint8 = crop.astype(np.uint8)

    gray = cv2.cvtColor(crop_uint8, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    patch_std = _compute_local_patch_std(gray, patch_size=16)
    noise_est = _compute_noise_estimate(gray)

    # 2D FFT High-frequency energy
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = 20 * np.log(np.abs(f_shift) + 1e-6)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4
    y_g, x_g = np.ogrid[:h, :w]
    hf_mask = np.sqrt((x_g - cx) ** 2 + (y_g - cy) ** 2) > radius
    freq_energy = float(np.mean(magnitude[hf_mask]))

    ela = _compute_ela_score(crop_uint8)
    ycrcb = cv2.cvtColor(crop_uint8, cv2.COLOR_BGR2YCrCb)
    chroma_disp = float((np.std(ycrcb[:, :, 1]) + np.std(ycrcb[:, :, 2])) / 2.0)

    return {
        'laplacian_variance': round(lap_var, 2),
        'local_patch_std': round(patch_std, 2),
        'noise_estimate': round(noise_est, 2),
        'frequency_energy': round(freq_energy, 2),
        'ela_score': round(ela, 2),
        'color_disparity': round(chroma_disp, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Multi-Face & Multi-Frame Prediction Engine
# ─────────────────────────────────────────────────────────────────────────────

def predict_deepfake(face_crops, media_path=None, media_type="IMAGE", face_metadata_list=None):
    """
    Evaluates face crops using the genuine SigLIP ONNX deepfake classification model.
    Classifies content into REAL, FAKE, or UNCERTAIN.
    
    Class mapping confirmed for pranjal-pravesh/prithivML-deepfake-detector-model-v1:
      Index 0 -> FAKE (Manipulated)
      Index 1 -> REAL (Authentic)
    """
    total_faces = len(face_crops) if face_crops else 0

    if total_faces == 0:
        return {
            'prediction': 'UNCERTAIN',
            'confidence': 0.0,
            'real_probability': 0.0,
            'fake_probability': 0.0,
            'analysis_status': 'NO_FACE',
            'model_available': False,
            'is_heuristic': False,
            'engine_used': 'None (No Face Crops)',
            'faces_count': 0,
            'faces_analysis': [],
            'explanation': 'No detectable human face found in uploaded media.',
            'forensic_details': {}
        }

    session, model_info = load_deepfake_model()

    # ═════════════════════════════════════════════════════════════════════════
    # CASE 1: Pretrained Machine Learning Model IS Available
    # ═════════════════════════════════════════════════════════════════════════
    if session is not None and model_info.get('available'):
        faces_analysis = []
        fake_probabilities = []
        real_probabilities = []

        input_name = model_info.get('input_name', 'pixel_values')
        output_name = model_info.get('output_name', 'logits')

        for idx, crop in enumerate(face_crops):
            face_id = idx + 1
            meta = face_metadata_list[idx] if (face_metadata_list and idx < len(face_metadata_list)) else {}

            try:
                # 1. Preprocess crop with SigLIP normalization
                tensor = preprocess_for_siglip(crop)

                # 2. Run ONNX Runtime inference
                outputs = session.run([output_name], {input_name: tensor})
                logits = np.squeeze(outputs[0])

                # 3. Numerically stable Softmax calculation
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)

                # Confirmed class mapping: Index 0 = Fake, Index 1 = Real
                fake_p = float(probs[0])
                real_p = float(probs[1])

                # Apply uncertainty region [0.45, 0.55]
                if 0.45 <= fake_p <= 0.55:
                    face_verdict = 'UNCERTAIN'
                    face_conf = round(abs(fake_p - 0.5) * 200.0, 2)
                elif fake_p > 0.55:
                    face_verdict = 'FAKE'
                    face_conf = round(fake_p * 100.0, 2)
                else:
                    face_verdict = 'REAL'
                    face_conf = round(real_p * 100.0, 2)

                fake_probabilities.append(fake_p)
                real_probabilities.append(real_p)

            except Exception as e:
                print(f"[Model Inference Error on Face #{face_id}]: {e}")
                face_verdict = 'UNCERTAIN'
                face_conf = 0.0
                real_p, fake_p = 0.5, 0.5
                fake_probabilities.append(0.5)
                real_probabilities.append(0.5)

            # Compute supporting forensic measurements
            forensic_sig = calculate_forensic_metrics_for_crop(crop)

            faces_analysis.append({
                'face_id': face_id,
                'label': f"Face #{face_id}",
                'model_prediction': face_verdict,
                'confidence': face_conf,
                'real_probability': round(real_p, 4),
                'fake_probability': round(fake_p, 4),
                'bbox': meta.get('bbox', [0, 0, 0, 0]),
                'crop_path': meta.get('crop_path', None),
                'frame_number': meta.get('frame_number', 1),
                'frame_timestamp': meta.get('frame_timestamp', 0.0),
                'forensic_signals': forensic_sig,
            })

        # Aggregation Logic
        # Video aggregation: accounts for temporal frame correlation
        if media_type == "VIDEO":
            q80_fake = float(np.percentile(fake_probabilities, 80)) if fake_probabilities else 0.5
            mean_fake = float(np.mean(fake_probabilities)) if fake_probabilities else 0.5
            aggregated_fake_p = round(0.6 * q80_fake + 0.4 * mean_fake, 4)
            aggregated_real_p = round(1.0 - aggregated_fake_p, 4)

            if 0.45 <= aggregated_fake_p <= 0.55:
                overall_verdict = 'UNCERTAIN'
                overall_conf = round(abs(aggregated_fake_p - 0.5) * 200.0, 2)
                explanation = (
                    "Video analysis indicates borderline facial indicators across sampled keyframes. "
                    "Note: Image model evaluates frames individually; temporal motion is not modeled."
                )
            elif aggregated_fake_p > 0.55:
                overall_verdict = 'FAKE'
                overall_conf = round(aggregated_fake_p * 100.0, 2)
                explanation = (
                    f"Frame-by-frame analysis detected persistent manipulation artifacts (80th percentile score: {q80_fake:.2f}). "
                    "Note: The model score represents an algorithmic estimation and is not an infallible proof of manipulation."
                )
            else:
                overall_verdict = 'REAL'
                overall_conf = round(aggregated_real_p * 100.0, 2)
                explanation = (
                    "Consistent authentic facial biometric characteristics observed across sampled video keyframes. "
                    "Note: Prediction reflects algorithmic confidence rather than guaranteed authenticity."
                )

        else:
            # Image aggregation: inspect both face crops and global media context
            crop_fake_p = max(fake_probabilities) if fake_probabilities else 0.5
            mean_real_p = float(np.mean(real_probabilities)) if real_probabilities else 0.5

            # Multi-scale inspection: evaluate full image frame for global AI generation artifacts
            full_frame_fake_p = None
            if media_type == "IMAGE" and media_path and os.path.isfile(media_path):
                try:
                    full_img = cv2.imread(media_path)
                    if full_img is not None and full_img.size > 0:
                        t_full = preprocess_for_siglip(full_img)
                        out_full = session.run([output_name], {input_name: t_full})
                        l_full = np.squeeze(out_full[0])
                        e_full = np.exp(l_full - np.max(l_full))
                        p_full = e_full / np.sum(e_full)
                        full_frame_fake_p = float(p_full[0])
                except Exception as e:
                    print(f"[Full Frame SigLIP Inspection Error]: {e}")

            # Combine face crop and full-frame signals
            # If the full frame displays high synthetic artifacts (common in full AI portraits)
            # or any face crop shows manipulation (common in face swaps)
            if full_frame_fake_p is not None and full_frame_fake_p > 0.65:
                # Full frame clearly indicates synthetic generation
                aggregated_fake_p = round(max(crop_fake_p, full_frame_fake_p), 4)
                aggregated_real_p = round(1.0 - aggregated_fake_p, 4)
                overall_verdict = 'FAKE'
                overall_conf = round(aggregated_fake_p * 100.0, 2)
                explanation = (
                    "Global image synthesis artifacts detected in full-frame analysis. "
                    f"Overall synthetic probability: {aggregated_fake_p:.1%}. "
                    "Characteristics are consistent with AI-generated media."
                )
            elif crop_fake_p > 0.55:
                overall_verdict = 'FAKE'
                overall_conf = round(crop_fake_p * 100.0, 2)
                aggregated_fake_p = round(crop_fake_p, 4)
                aggregated_real_p = round(1.0 - aggregated_fake_p, 4)
                explanation = (
                    "Deepfake generation artifacts detected in analyzed facial content. "
                    "Model score reflects learned pattern probability, not a guaranteed verdict."
                )
            elif crop_fake_p < 0.45 and (full_frame_fake_p is None or full_frame_fake_p < 0.45):
                overall_verdict = 'REAL'
                overall_conf = round(mean_real_p * 100.0, 2)
                aggregated_real_p = round(mean_real_p, 4)
                aggregated_fake_p = round(1.0 - aggregated_real_p, 4)
                explanation = (
                    "Authentic facial features and natural biometric texture observed. "
                    "Model score indicates statistical likelihood of authenticity."
                )
            else:
                overall_verdict = 'UNCERTAIN'
                max_fake_val = max(crop_fake_p, full_frame_fake_p if full_frame_fake_p is not None else crop_fake_p)
                overall_conf = round(abs(max_fake_val - 0.5) * 200.0, 2)
                aggregated_fake_p = round(max_fake_val, 4)
                aggregated_real_p = round(1.0 - aggregated_fake_p, 4)
                explanation = (
                    "Model outputs fall within the borderline ambiguity threshold. "
                    "Evidence is insufficient to draw an authoritative Real or Fake conclusion."
                )

        return {
            'prediction': overall_verdict,
            'confidence': overall_conf,
            'real_probability': aggregated_real_p,
            'fake_probability': aggregated_fake_p,
            'analysis_status': 'COMPLETED',
            'model_available': True,
            'is_heuristic': False,
            'engine_used': f"prithivML SigLIP ONNX ({model_info.get('filename')})",
            'faces_count': total_faces,
            'faces_analysis': faces_analysis,
            'explanation': explanation,
            'forensic_details': {
                'engine': f"Pretrained SigLIP ONNX ({model_info.get('runtime')})",
                'model_file': model_info.get('filename'),
                'model_path': model_info.get('path'),
                'normalization': model_info.get('normalization'),
                'input_tensor': model_info.get('input_name'),
                'output_tensor': model_info.get('output_name'),
                'faces_analyzed_count': total_faces,
                'is_heuristic': False,
                'model_available': True,
                'faces_analysis': faces_analysis,
            }
        }

    # ═════════════════════════════════════════════════════════════════════════
    # CASE 2: Classifier IS UNAVAILABLE
    # Never return a hardcoded Real or Fake result when model is missing.
    # ═════════════════════════════════════════════════════════════════════════
    faces_analysis = []
    for idx, crop in enumerate(face_crops):
        face_id = idx + 1
        meta = face_metadata_list[idx] if (face_metadata_list and idx < len(face_metadata_list)) else {}
        sig = calculate_forensic_metrics_for_crop(crop)
        faces_analysis.append({
            'face_id': face_id,
            'label': f"Face #{face_id}",
            'model_prediction': 'UNCERTAIN (Model Unavailable)',
            'confidence': 0.0,
            'real_probability': 0.0,
            'fake_probability': 0.0,
            'bbox': meta.get('bbox', [0, 0, 0, 0]),
            'crop_path': meta.get('crop_path', None),
            'frame_number': meta.get('frame_number', 1),
            'frame_timestamp': meta.get('frame_timestamp', 0.0),
            'forensic_signals': sig,
        })

    return {
        'prediction': 'UNCERTAIN',
        'confidence': 0.0,
        'real_probability': 0.0,
        'fake_probability': 0.0,
        'analysis_status': 'MODEL_UNAVAILABLE',
        'model_available': False,
        'is_heuristic': True,
        'engine_used': 'Forensic Telemetry (Model Unavailable)',
        'faces_count': total_faces,
        'faces_analysis': faces_analysis,
        'explanation': (
            'The pretrained ONNX classifier (deepfake_detector_v1.onnx) was not found in model/ directory. '
            'Inference was not performed. Only supporting forensic indicators are shown.'
        ),
        'forensic_details': {
            'engine': 'Forensic Telemetry Engine (Model Unavailable)',
            'faces_analyzed_count': total_faces,
            'is_heuristic': True,
            'model_available': False,
            'faces_analysis': faces_analysis,
        }
    }
