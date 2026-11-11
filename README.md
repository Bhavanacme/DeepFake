# Deepfake Detection System

A final-year Diploma CSE project implementing an AI-powered media authenticity verification system built with Django, OpenCV, and computer vision forensic analysis.

---

## ⚠ Model Status

**A trained deepfake classifier (deepfake_model.h5 or deepfake_model.onnx) is NOT included** due to dataset licensing and file size constraints.

Without a trained model, the system operates in **forensic heuristic mode** — it detects faces and reports computer-vision measurements (Laplacian variance, ELA, frequency analysis) but **cannot issue a reliable REAL/FAKE verdict**. The UI clearly labels all heuristic results as "Analysis Inconclusive" to prevent misleading outputs.

To enable full AI classification, follow the training steps below.

---

## Technology Stack

| Component       | Version         |
|-----------------|-----------------|
| Python          | 3.14.6          |
| Django          | 6.1.1           |
| OpenCV          | 5.0.0 (+ YuNet) |
| Pillow          | 12.2.0          |
| NumPy           | 2.5.0           |
| ReportLab       | 5.0.1 (PDF)     |
| Matplotlib      | 3.11.0          |
| pandas          | 3.0.3           |
| **TensorFlow**  | **Not installed** (Python 3.14 incompatible — use Python 3.10/3.11 venv for training) |

---

## Installation & Setup

### 1. Prerequisites
- Python 3.14 (for running the Django server)
- All dependencies already installed (see pip list output)

### 2. Clone / Navigate to Project
```bash
cd "c:\Users\User\OneDrive\Desktop\Deepfake Project"
```

### 3. Run Migrations
```bash
python manage.py migrate
```

### 4. Collect Static Files
```bash
python manage.py collectstatic --noinput
```

### 5. Create Admin User (optional)
```bash
python manage.py createsuperuser
```

### 6. Start Development Server
```bash
python manage.py runserver
```

### 7. Open in Browser
Visit: **http://127.0.0.1:8000**

---

## Training a Model

> **Note:** TensorFlow does not support Python 3.14 yet. Use a Python 3.10 or 3.11 virtual environment for training.

### Step 1: Prepare Dataset
Create this folder structure:
```
dataset/
    train/
        real/    <- Authentic photographs (cropped faces)
        fake/    <- AI-generated / face-swapped images
    val/
        real/
        fake/
```

#### Recommended Datasets:
- [140k Real and Fake Faces (Kaggle)](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- [FaceForensics++ (academic)](https://github.com/ondyari/FaceForensics)
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)

### Step 2: Install TensorFlow (in Python 3.10/3.11 venv)
```bash
pip install tensorflow tf2onnx
```

### Step 3: Train
```bash
python train_model.py --data_dir dataset/ --epochs 15 --batch_size 32
```

This produces:
- `model/deepfake_model.h5` — Keras model
- `model/deepfake_model.onnx` — ONNX model (usable on Python 3.14 without TensorFlow!)
- `model/class_indices.json` — Class label mapping

### Step 4: Copy ONNX Model Back
Place `model/deepfake_model.onnx` in the project's `model/` folder. The app will auto-detect and use it via OpenCV DNN.

---

## Evaluating the Model

```bash
python evaluate_model.py --test_dir test_data/ --threshold 0.5
```

**Requires:** `test_data/real/` and `test_data/fake/` folders with labeled images.

**Output (saved to `model/evaluation/`):**
- `evaluation_report.txt` — Accuracy, Precision, Recall, F1, Confusion Matrix
- `confusion_matrix.png` — Visual confusion matrix
- `class_mapping.json` — Confirmed class label mapping

---

## Class Label Mapping

| Model Output     | Meaning |
|-----------------|---------|
| `sigmoid = 0.0 → 0.49` | REAL (probability of fake is low) |
| `sigmoid = 0.5 → 1.0`  | FAKE (probability of fake is high) |
| `softmax[0]`            | P(REAL) |
| `softmax[1]`            | P(FAKE) |

**Class indices (train_model.py enforces this):**
```
class_names=["real", "fake"]   → index 0 = REAL, index 1 = FAKE
```

---

## Pages & Features

| URL                  | Description              | Auth Required |
|---------------------|--------------------------|---------------|
| `/`                  | Home page                | No            |
| `/register/`         | User registration        | No (guest)    |
| `/login/`            | Login                    | No (guest)    |
| `/logout/`           | Logout                   | Yes           |
| `/dashboard/`        | User dashboard           | Yes           |
| `/detect/`           | Upload & analyze media   | Yes           |
| `/result/<id>/`      | Detection result         | Yes (owner)   |
| `/history/`          | Analysis history         | Yes           |
| `/history/<id>/`     | Detailed scan view       | Yes (owner)   |
| `/report/<id>/`      | Download PDF report      | Yes (owner)   |
| `/profile/`          | User profile             | Yes           |
| `/admin-dashboard/`  | Admin statistics         | Staff only    |
| `/admin/`            | Django admin site        | Staff only    |
| `/about/`            | About page               | No            |
| `/how-it-works/`     | Pipeline explanation     | No            |
| `/contact/`          | Contact form             | No            |

---

## Detection Pipeline

1. **Upload** — File validated (type, size, integrity)
2. **Face Detection** — YuNet ONNX model (OpenCV 5 FaceDetectorYN)
3. **Preprocessing** — Face crop, resize to 224×224, normalize [0,1]
4. **Inference** — Trained CNN classifier (ONNX or Keras)
5. **Result** — REAL / FAKE / INCONCLUSIVE verdict + forensic indicators
6. **Report** — PDF report via ReportLab

---

## Detection Limitations

- **No model = No verdict.** Without `deepfake_model.onnx`, results are heuristic only.
- **Face required.** Images without a detectable face return "No Face Detected".
- **Not an audio deepfake detector.** Only visual media is analyzed.
- **Not calibrated.** Confidence scores reflect raw model sigmoid output, not calibrated probabilities.
- **Image quality matters.** Low-resolution, heavily compressed, or partially visible faces reduce accuracy.
- **GAN cutoff heuristics are unreliable** for modern diffusion-model images.

---

## Project Structure

```
Deepfake Project/
├── manage.py
├── deepfake_project/        # Django configuration
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── detection/               # Main detection app
│   ├── ml_model.py          # Model loading & inference
│   ├── preprocessing.py     # Face detection & image preprocessing
│   ├── views.py             # Django views
│   ├── models.py            # DetectionResult database model
│   ├── urls.py              # URL patterns
│   ├── forms.py             # Upload form
│   └── utils.py             # PDF report generation
├── users/                   # Authentication app
│   ├── views.py
│   ├── models.py            # UserProfile
│   ├── forms.py
│   └── urls.py
├── templates/               # HTML templates
│   ├── base.html
│   └── detection/, users/
├── static/                  # CSS, JS, images
├── media/                   # User uploads & processed faces
├── model/                   # AI model files (place here)
│   └── face_detection_yunet_2023mar.onnx  ✅ Present
├── train_model.py           # Training script (needs TF + dataset)
├── evaluate_model.py        # Model evaluation & label verification
└── requirements.txt
```

---

## Security Notes

- CSRF protection enabled on all POST forms
- Login required on all private views
- Ownership checks prevent users from accessing other users' data
- File type and size validated both client-side and server-side
- `DEBUG = True` — **Set to False in production**
- `SECRET_KEY` — Replace with environment variable in production

---

## Known Issues & Roadmap

| Status | Issue |
|--------|-------|
| ✅ Fixed | Missing CSS classes (`stat-icon-wrapper`, `pipeline-step-card`, etc.) |
| ✅ Fixed | Hardcoded "98.2% Accurate" in hero — replaced with real platform stats |
| ✅ Fixed | Home page exposed other users' data to guests — now private |
| ✅ Fixed | No Register link in navbar for guests |
| ✅ Fixed | Timezone was UTC — now set to Asia/Kolkata (IST) |
| ✅ Fixed | Upload card on home had negative margin without hero context |
| ✅ Added | Reduced-motion accessibility CSS |
| ✅ Added | Focus-visible keyboard navigation styles |
| ✅ Added | evaluate_model.py verification script |
| ⏳ Pending | Train & provide deepfake_model.onnx |
| ⏳ Pending | Collect labeled test dataset for evaluation |
| ⏳ Pending | Install TensorFlow in Python 3.10/3.11 venv for training |

---

## Developed By
Diploma CSE Final-Year Project  
Built with Django 6.1, OpenCV 5, Bootstrap 5, Python 3.14
