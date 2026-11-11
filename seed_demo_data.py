import os
import io
import django
import cv2
import numpy as np
from PIL import Image

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_project.settings')
django.setup()

from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from detection.models import DetectionResult
from users.models import UserProfile


def seed_data():
    print("Seeding demo users and initial detection samples...")

    # 1. Ensure Student User exists
    student, created = User.objects.get_or_create(
        username='student',
        defaults={
            'email': 'student@deepfake.edu',
            'first_name': 'Alex',
            'last_name': 'Morgan'
        }
    )
    student.set_password('student123')
    student.save()

    profile, _ = UserProfile.objects.get_or_create(user=student)
    profile.organization = "Diploma CSE - Final Year"
    profile.bio = "Final Year CSE student researching deep learning biometric security and synthetic media forensics."
    profile.save()

    # 2. Generate a sample face image with bounding box
    faces_dir = os.path.join(settings.MEDIA_ROOT, 'processed', 'faces')
    os.makedirs(faces_dir, exist_ok=True)

    # Draw synthetic sample face 1 (Authentic)
    img_real = np.full((300, 300, 3), 245, dtype=np.uint8)
    cv2.circle(img_real, (150, 150), 80, (200, 180, 160), -1) # face base
    cv2.circle(img_real, (120, 130), 10, (80, 60, 50), -1)   # left eye
    cv2.circle(img_real, (180, 130), 10, (80, 60, 50), -1)   # right eye
    cv2.ellipse(img_real, (150, 180), (35, 15), 0, 0, 180, (50, 50, 180), 3) # smile
    # Add biometric detection box
    cv2.rectangle(img_real, (60, 60), (240, 240), (255, 180, 0), 2)
    cv2.putText(img_real, "Face #1 [Verified]", (60, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)
    
    real_face_rel = "processed/faces/demo_real_face.jpg"
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, real_face_rel), img_real)

    # Draw synthetic sample face 2 (Deepfake with boundary artifacts)
    img_fake = np.full((300, 300, 3), 240, dtype=np.uint8)
    cv2.circle(img_fake, (150, 150), 80, (190, 175, 170), -1)
    cv2.circle(img_fake, (122, 132), 12, (50, 40, 40), -1)
    cv2.circle(img_fake, (178, 132), 12, (50, 40, 40), -1)
    cv2.ellipse(img_fake, (150, 180), (30, 12), 0, 0, 180, (40, 40, 170), 3)
    # Synthetic boundary seam
    cv2.circle(img_fake, (150, 150), 82, (160, 150, 210), 1)
    cv2.rectangle(img_fake, (60, 60), (240, 240), (0, 0, 220), 2)
    cv2.putText(img_fake, "Deepfake Warp [Alert]", (60, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)
    
    fake_face_rel = "processed/faces/demo_fake_face.jpg"
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, fake_face_rel), img_fake)

    # 3. Insert or update demo DetectionResult records
    if not DetectionResult.objects.filter(user=student).exists():
        DetectionResult.objects.create(
            user=student,
            file_name="sample_video_interview.mp4",
            file_type="VIDEO",
            file_size_mb=14.25,
            result="FAKE",
            confidence=92.45,
            fake_probability=0.9245,
            real_probability=0.0755,
            face_detected=True,
            faces_count=1,
            frames_analyzed=15,
            extracted_face=fake_face_rel,
            model_name="CNN Deepfake Classifier v2.1",
            forensic_details={
                'engine': 'OpenCV Spectral & Facial Artifact Forensic Analyzer',
                'laplacian_sharpness_var': 64.3,
                'high_frequency_energy': 48.7,
                'chroma_space_disparity': 8.9,
                'analyzed_samples': 15,
                'notes': 'High probability of autoencoder latent swap blending detected.'
            },
            processing_status="COMPLETED"
        )

        DetectionResult.objects.create(
            user=student,
            file_name="id_portrait_photo.jpg",
            file_type="IMAGE",
            file_size_mb=2.10,
            result="REAL",
            confidence=94.80,
            fake_probability=0.0520,
            real_probability=0.9480,
            face_detected=True,
            faces_count=1,
            frames_analyzed=1,
            extracted_face=real_face_rel,
            model_name="CNN Deepfake Classifier v2.1",
            forensic_details={
                'engine': 'OpenCV Spectral & Facial Artifact Forensic Analyzer',
                'laplacian_sharpness_var': 188.5,
                'high_frequency_energy': 114.2,
                'chroma_space_disparity': 24.6,
                'analyzed_samples': 1,
                'notes': 'Normal natural skin pore texture variance and optical noise observed.'
            },
            processing_status="COMPLETED"
        )

        DetectionResult.objects.create(
            user=student,
            file_name="synthetic_avatar_profile.png",
            file_type="IMAGE",
            file_size_mb=1.65,
            result="FAKE",
            confidence=88.10,
            fake_probability=0.8810,
            real_probability=0.1190,
            face_detected=True,
            faces_count=1,
            frames_analyzed=1,
            extracted_face=fake_face_rel,
            model_name="CNN Deepfake Classifier v2.1",
            forensic_details={
                'engine': 'OpenCV Spectral & Facial Artifact Forensic Analyzer',
                'laplacian_sharpness_var': 71.0,
                'high_frequency_energy': 56.4,
                'chroma_space_disparity': 10.2,
                'analyzed_samples': 1,
                'notes': 'Generative model boundary checkerboard frequency artifacts observed.'
            },
            processing_status="COMPLETED"
        )

        DetectionResult.objects.create(
            user=student,
            file_name="scenic_landscape.jpg",
            file_type="IMAGE",
            file_size_mb=3.80,
            result="UNCERTAIN",
            confidence=0.0,
            fake_probability=0.0,
            real_probability=0.0,
            face_detected=False,
            faces_count=0,
            frames_analyzed=1,
            model_name="Haar Cascade Biometric Scanner",
            error_message="No recognizable human face was detected in this image. Please upload a clear photo with a visible face.",
            processing_status="NO_FACE"
        )
        print("Created 4 sample detection records successfully!")
    else:
        print("Demo records already exist.")

if __name__ == '__main__':
    seed_data()
