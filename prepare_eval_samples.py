"""
prepare_eval_samples.py
Generates sample benchmark images for reproducible evaluation testing.
"""
import os
import cv2
import numpy as np

def generate_faces(output_dir="test_dataset", count=10):
    os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)
    
    rng = np.random.RandomState(101)

    # 1. Real samples (with sensor noise, natural gradient, high-frequency pores)
    for i in range(count):
        img = np.full((224, 224, 3), 180, dtype=np.uint8)
        # Background
        cv2.circle(img, (112, 112), 70, (190, 205, 220), -1)
        # Facial features
        cv2.circle(img, (90, 95), 10, (50, 40, 30), -1)
        cv2.circle(img, (134, 95), 10, (50, 40, 30), -1)
        cv2.ellipse(img, (112, 145), (20, 10), 0, 0, 180, (60, 60, 120), 3)
        # Camera sensor noise
        noise = rng.normal(0, 14, (224, 224, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "real", f"real_face_{i+1:02d}.jpg"), img)

    # 2. Fake samples (smoothed, blended boundary, generative latent artifacts)
    for i in range(count):
        img = np.full((224, 224, 3), 180, dtype=np.uint8)
        cv2.circle(img, (112, 112), 70, (190, 205, 220), -1)
        cv2.circle(img, (90, 95), 10, (50, 40, 30), -1)
        cv2.circle(img, (134, 95), 10, (50, 40, 30), -1)
        cv2.ellipse(img, (112, 145), (20, 10), 0, 0, 180, (60, 60, 120), 3)
        # Smooth / blur boundary typical of generative models
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        # Add high-frequency edge perturbation at boundary
        cv2.circle(img, (112, 112), 70, (230, 180, 170), 2)
        cv2.imwrite(os.path.join(output_dir, "fake", f"fake_face_{i+1:02d}.jpg"), img)

    print(f"Generated {count} real and {count} fake benchmark samples in '{output_dir}'.")

if __name__ == '__main__':
    generate_faces()
