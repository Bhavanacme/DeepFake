# Deepfake Detection System — Project Report & Viva-Voce Guide
> **Diploma in Computer Science & Engineering (CSE) — Final Year Project**  
> **Topic:** AI-Based Deepfake Detection & Media Authenticity Verification System  
> **Prepared for:** Final Project Demonstration, Viva-Voce, and Project Review

---

## 1. Project Abstract

With the explosive growth of artificial intelligence, deep learning generative algorithms (such as Generative Adversarial Networks — GANs, Variational Autoencoders — VAEs, and Diffusion models) can generate hyper-realistic synthetic human images and videos, commonly known as **Deepfakes**. These manipulated media pose severe threats to digital security, public trust, executive privacy, and the integrity of digital news.

The **Deepfake Detection System** is an enterprise-grade, full-stack web application designed and developed using **Python, Django, OpenCV, Bootstrap 5, and Deep Learning principles**. The system enables registered users to upload images and videos, automatically validates file integrity, extracts facial biometric contours using OpenCV Haar Cascade classifiers, and executes deep neural forensic analysis to predict whether the media is **REAL or FAKE/DEEPFAKE** with an exact confidence percentage. The platform features an interactive user dashboard, historical scan tracking, user profile management, a dedicated administrative control panel, and automated downloadable PDF forensic audit certificates.

---

## 2. Problem Statement & Need

### The Problem:
- **Visual Deception:** Modern deepfake tools (e.g., DeepFaceLab, FaceSwap, Wav2Lip) make visual discrimination impossible for the naked human eye.
- **Cybersecurity & Impersonation:** Deepfakes are increasingly exploited for biometric KYC fraud, digital identity theft, and corporate executive spoofing.
- **Lack of Accessible Verification Tools:** Most digital forensics tools exist only as command-line scripts or academic notebooks, inaccessible to journalists, students, and normal users.

### The Solution:
- A user-friendly, responsive, browser-based verification platform with a clean **Blue & White** design.
- Automated end-to-end processing pipeline: from drag-and-drop file upload to instant PDF audit certificate download.
- Extensible AI integration allowing plug-and-play placement of trained convolutional neural networks (`deepfake_model.h5`).

---

## 3. Technology Stack & Justification

| Component | Technology | Technical Justification |
|---|---|---|
| **Backend Framework** | **Django 6.1 (Python)** | High-security built-in authentication, ORM database abstraction, robust CSRF protection, and rapid modular scalability. |
| **Frontend UI** | **HTML5, CSS3, Bootstrap 5.3, JavaScript** | Responsive grid system, mobile-first design, interactive animations, drag-and-drop events, and zero dependency overhead. |
| **Visual Charts** | **Chart.js** | Client-side responsive doughnut and pie chart rendering for forensic telemetry and dashboard analytics. |
| **Image & Video Processing** | **OpenCV 5.0 (`cv2`)** | Industry-standard computer vision library for fast frame extraction, color space conversions, and image decodings. |
| **Face Detection** | **OpenCV Haar Cascade** | Proven, lightweight real-time facial landmark localization using `haarcascade_frontalface_default.xml`. |
| **Deep Learning & Forensics** | **TensorFlow/Keras + CV Spectral Engine** | Pluggable CNN `.h5` model loader paired with 2D Fast Fourier Transform (FFT) and Laplacian variance texture forensics. |
| **Database** | **SQLite (MySQL Ready)** | Zero-configuration file database for development; standard Django ORM enables seamless switching to MySQL for production. |
| **PDF Reporting** | **ReportLab 5.0** | Server-side programmatic PDF generation with custom typography, tables, and embedded facial biometrics. |

---

## 4. System Architecture & Detection Pipeline

```text
       ┌────────────────────────┐
       │      User Upload       │ (JPG, PNG, MP4, AVI, MOV)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Server File Validation │ (Check size <= 50MB, MIME, headers)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Media Preprocessing    │ 
       │ - Video Keyframe Split │ (10-15 evenly sampled frames)
       │ - Image Resize & Norm  │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ OpenCV Face Detection  │
       │ (Haar Cascade XML)     │
       └─────┬────────────┬─────┘
             │            │
       [Face Found]    [No Face Found]
             │            │
             │            └───► Set status = NO_FACE (Friendly Warning)
             ▼
       ┌────────────────────────────────────────┐
       │ AI Model & Forensics Layer             │
       │  a) If deepfake_model.h5 exists:       │
       │     --> CNN Feature Map Inference      │
       │  b) If H5 not loaded:                  │
       │     --> Laplacian Sharpness Variance   │
       │     --> 2D FFT Spectral Energy         │
       │     --> YCbCr Chrominance Disparity    │
       └────────────────────┬───────────────────┘
                            │
                            ▼
       ┌────────────────────────────────────────┐
       │ Prediction & Confidence Scoring        │
       │ - REAL / FAKE Verdict Badge            │
       │ - Confidence: 0 - 100%                 │
       │ - Save to Database (DetectionResult)   │
       └────────────────────┬───────────────────┘
                            │
                            ▼
       ┌────────────────────────────────────────┐
       │ Output: User Dashboard, Scan Details,  │
       │ and Downloadable ReportLab PDF Report  │
       └────────────────────────────────────────┘
```

---

## 5. Detailed Module Descriptions

### Module 1: User Management & Authentication (`users/`)
- Handles user registration, encrypted password storage with Django's PBKDF2 hashing, and multi-field login (supports username or email).
- Extends standard `User` with `UserProfile` for profile photos, bio, and student organization affiliation.

### Module 2: File Ingestion & Validation (`detection/forms.py` & `preprocessing.py`)
- Restricts uploads to permitted extensions: `.jpg`, `.jpeg`, `.png`, `.mp4`, `.avi`, `.mov`, `.mkv`.
- Enforces 50 MB maximum file threshold.
- Validates file headers and PIL integrity to prevent corrupted file injections.

### Module 3: Face Detection & Biometric Cropping (`detection/preprocessing.py`)
- Standardizes images into grayscale with histogram equalization for balanced contrast.
- Utilizes `cv2.CascadeClassifier(haarcascade_frontalface_default.xml)` with multi-scale scanning (`scaleFactor=1.1, minNeighbors=5`).
- Crops detected regions of interest (ROI) with 15% contextual boundary margins.
- Annotates original media with sleek biometric bounding boxes and corner bracket coordinates.
- For video inputs, samples frames evenly across playback duration to capture temporal consistency.

### Module 4: Pluggable Deep Learning & Forensic Engine (`detection/ml_model.py`)
- **Trained Model Architecture:** If `model/deepfake_model.h5` is placed in the project, the system loads it via Keras (`tf.keras.models.load_model`) and evaluates batch tensor inputs.
- **Forensic Spectral Engine:** When running in forensic mode, it calculates:
  $$\text{Laplacian Variance} = \text{Var}\left(\nabla^2 I\right) = \frac{1}{N}\sum (L_{x,y} - \mu)^2$$
  *(Low variance indicates synthetic GAN smoothing; high variance indicates natural pores and optical camera noise).*
  $$\text{2D FFT Energy} = \frac{1}{|M|}\sum_{(u,v) \in M} \log(1 + |F(u,v)|)$$
  *(Measures high-frequency cutoff characteristics in the frequency domain).*

### Module 5: Reporting & PDF Verification (`detection/utils.py`)
- Generates a PDF **Certificate of Analysis** containing a unique ID (`DFS-XXXXXX`), submitter name, detection timestamp, biometric crop, and verification disclaimer.

### Module 6: System Administration Dashboard (`templates/detection/admin_dashboard.html`)
- Provides superusers with real-time statistics: total users, uploads, verdict distribution, storage footprint, and quick moderation links.

---

## 6. Database Schema (Entity-Relationship)

```
┌──────────────────────┐         1:1         ┌──────────────────────┐
│     auth_user        ├─────────────────────┤  users_userprofile   │
├──────────────────────┤                     ├──────────────────────┤
│ id (PK)              │                     │ id (PK)              │
│ username             │                     │ user_id (FK)         │
│ email                │                     │ avatar               │
│ first_name           │                     │ organization         │
│ last_name            │                     │ bio                  │
│ password (hashed)    │                     │ phone_number         │
│ date_joined          │                     └──────────────────────┘
└──────────┬───────────┘
           │ 1
           │
           │ *
           ▼
┌────────────────────────────────────────┐
│      detection_detectionresult         │
├────────────────────────────────────────┤
│ id (PK)                                │
│ user_id (FK to auth_user)              │
│ file (FileField)                       │
│ file_name (CharField)                 │
│ file_type (IMAGE / VIDEO)              │
│ file_size_mb (FloatField)              │
│ result (REAL / FAKE / UNCERTAIN)       │
│ confidence (FloatField: 0 - 100%)      │
│ real_probability (FloatField: 0.0-1.0) │
│ fake_probability (FloatField: 0.0-1.0) │
│ face_detected (BooleanField)           │
│ faces_count (IntegerField)             │
│ extracted_face (ImageField)            │
│ frames_analyzed (IntegerField)         │
│ model_name (CharField)                 │
│ forensic_details (JSONField)           │
│ processing_status (CharField)          │
│ error_message (TextField)              │
│ detected_at (DateTimeField)            │
└────────────────────────────────────────┘
```

---

## 7. Viva-Voce Questions & Model Answers

### Q1: What is the main objective of your project?
> **Answer:** "Our project, the Deepfake Detection System, is designed to analyze uploaded images and videos and accurately classify whether they are authentic or synthetic/manipulated deepfakes using deep learning and computer vision biometrics. It provides confidence percentages, detailed forensic telemetry, user scan histories, and downloadable PDF verification reports."

### Q2: Why did you choose Django for this project instead of Flask or PHP?
> **Answer:** "Django was chosen because it is an enterprise-level Python framework following the Model-View-Template (MVT) pattern. It has built-in secure user authentication, automatic CSRF protection, an ORM for structured database management, and native compatibility with Python's data science and computer vision libraries like OpenCV, NumPy, and TensorFlow."

### Q3: How do you extract and detect faces from the uploaded files?
> **Answer:** "We use OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`). The media is first standardized and converted to grayscale with histogram equalization. The multi-scale detection algorithm slides windows across the image to locate facial edge and line features. The primary face is cropped with a 15% border margin and resized to $224 \times 224$ pixels normalized for model evaluation."

### Q4: What happens if a user uploads an image without a human face?
> **Answer:** "The system handles it gracefully without crashing. If OpenCV returns zero face bounding boxes, the system records `processing_status = 'NO_FACE'` and displays a user-friendly message: *'No recognizable face was detected. Please upload another image or video containing a clear face.'* This prevents false classifications on scenery or non-facial objects."

### Q5: How does your system process video files?
> **Answer:** "Using `cv2.VideoCapture()`, the video is opened and its frame rate and frame count are read. We sample 10 to 15 evenly distributed keyframes across the playback timeline. Each frame undergoes facial biometric detection, capturing temporal consistency and inter-frame artifacts, and the best annotated keyframe is saved for visual confirmation."

### Q6: What is Laplacian Texture Variance and why is it used?
> **Answer:** "The Laplacian operator calculates the 2nd spatial derivative of an image, emphasizing rapid intensity changes such as edges and textures. Real human skin exhibits natural pores, wrinkles, and fine optical camera sensor noise, resulting in higher variance. Deepfakes generated by Autoencoders or GANs often suffer from boundary blending or synthetic skin-smoothing, which produces abnormally low Laplacian variance."

### Q7: How can an examiner or developer plug in their own trained deep learning model?
> **Answer:** "We engineered a pluggable model interface in `detection/ml_model.py`. Any trained Keras/TensorFlow model can simply be saved as `model/deepfake_model.h5`. When TensorFlow is installed and this file is present, the application automatically detects and loads it on startup without requiring changes to the Django views or database models."

### Q8: What security practices have been implemented?
> **Answer:**
> 1. All passwords are encrypted using Django's PBKDF2 hashing with SHA-256.
> 2. All POST forms require valid CSRF verification tokens.
> 3. Private scan records are restricted by user ID via `@login_required` decorators.
> 4. File uploads are validated by MIME type, extension, and a 50 MB maximum size limit.
> 5. Staff and admin privileges are enforced via `@staff_member_required`.

---

## 8. Summary for Project Examiners

The **Deepfake Detection System** successfully bridges academic machine learning research and real-world web deployment. It delivers a fast, responsive, and secure platform suitable for final-year Diploma CSE submission, viva-voce demonstration, and practical deployment in digital media verification.
