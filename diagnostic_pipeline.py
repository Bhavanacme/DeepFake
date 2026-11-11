"""
Diagnostic script — traces the forensic pipeline on synthetic and real images.
Run: python diagnostic_pipeline.py
"""
import os, sys
sys.path.insert(0, r'c:\Users\User\OneDrive\Desktop\Deepfake Project')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_project.settings')

import django
django.setup()

import cv2
import numpy as np


def measure_forensics(crop_uint8, label):
    gray = cv2.cvtColor(crop_uint8, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-6)
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 4
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    high_freq_mask = dist_from_center > radius
    high_freq_energy = float(np.mean(magnitude_spectrum[high_freq_mask]))

    ycrcb = cv2.cvtColor(crop_uint8, cv2.COLOR_BGR2YCrCb)
    cr_std = np.std(ycrcb[:, :, 1])
    cb_std = np.std(ycrcb[:, :, 2])
    color_disparity = float((cr_std + cb_std) / 2.0)

    print(f'\n--- {label} ---')
    print(f'  Laplacian variance : {lap_var:.2f}')
    print(f'  High-freq energy   : {high_freq_energy:.2f}')
    print(f'  Color disparity    : {color_disparity:.2f}')

    anomaly_factor = 0.0
    notes = []
    if lap_var < 80.0:
        anomaly_factor += 0.35
        notes.append('Laplacian<80 (+0.35)')
    elif lap_var > 950.0:
        anomaly_factor += 0.25
        notes.append('Laplacian>950 (+0.25)')
    if high_freq_energy < 85.0:
        anomaly_factor += 0.30
        notes.append('HF<85 (+0.30)')
    if color_disparity < 12.0:
        anomaly_factor += 0.25
        notes.append('Color<12 (+0.25)')

    base_fake_prob = 0.20 + (anomaly_factor * 0.70)
    fake_prob = float(np.clip(base_fake_prob, 0.05, 0.96))
    real_prob = 1.0 - fake_prob
    prediction = 'FAKE' if fake_prob >= 0.5 else 'REAL'
    confidence = fake_prob * 100 if fake_prob >= 0.5 else real_prob * 100

    print(f'  Anomaly triggers   : {notes if notes else "NONE"}')
    print(f'  Anomaly factor     : {anomaly_factor}')
    print(f'  Fake probability   : {fake_prob:.4f}  ({fake_prob*100:.1f}%)')
    print(f'  Real probability   : {real_prob:.4f}  ({real_prob*100:.1f}%)')
    print(f'  PREDICTION         : {prediction}  (confidence {confidence:.1f}%)')
    return lap_var, high_freq_energy, color_disparity, fake_prob


np.random.seed(42)

print('='*60)
print('FORENSIC PIPELINE DIAGNOSTIC')
print('='*60)

# Simulate AI-generated face
ai_face = np.zeros((224, 224, 3), dtype=np.uint8)
for c in range(3):
    base = np.linspace(160, 200, 224).reshape(1, 224).repeat(224, axis=0)
    noise = np.random.normal(0, 3, (224, 224))
    ai_face[:, :, c] = np.clip(base + noise, 0, 255).astype(np.uint8)
measure_forensics(ai_face, 'SIMULATED AI IMAGE (smooth, low noise)')

# Simulate real photo
real_face = np.zeros((224, 224, 3), dtype=np.uint8)
for c in range(3):
    base = np.linspace(140, 210, 224).reshape(1, 224).repeat(224, axis=0)
    noise = np.random.normal(0, 18, (224, 224))
    real_face[:, :, c] = np.clip(base + noise, 0, 255).astype(np.uint8)
measure_forensics(real_face, 'SIMULATED REAL PHOTO (textured, camera noise)')

print('\n' + '='*60)
print('ROOT CAUSE ANALYSIS')
print('='*60)
print('When NO thresholds trigger: base_fake_prob = 0.20 + 0 * 0.70 = 0.20')
print('=> fake=0.20, real=0.80 => PREDICTION: REAL at 80.0%')
print('=> This is exactly what the user reported.')
print()
print('The heuristics were designed for old-style face-swap deepfakes')
print('(blurry, low-frequency, chroma-inconsistent). Modern AI-generated')
print('images are sharp and colorful, so they score as "anomaly-free" => REAL.')
