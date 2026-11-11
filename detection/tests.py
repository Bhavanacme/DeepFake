"""
Comprehensive test suite for the AI & Deepfake Detection System.

Coverage:
  - Public page availability
  - Authentication & authorisation
  - File validation (upload)
  - Deepfake model discovery, loading, SigLIP preprocessing, class mapping
  - AI-generated image detector module (load, preprocess, class mapping, verdict logic)
  - Combined four-state verdict logic
  - Database models (DetectionResult, FaceDetectionItem, AnalysisReport)
  - History view filters (including ai_generated)
  - PDF report generation
  - User privacy isolation

Run:
    python manage.py test detection --verbosity=2
"""

import io
import os
import numpy as np
import cv2
from PIL import Image

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from .models import DetectionResult, FaceDetectionItem, AnalysisReport
from .preprocessing import validate_uploaded_media, process_image
from .ml_model import (
    find_deepfake_classifier,
    load_deepfake_model,
    predict_deepfake,
    preprocess_for_siglip,
    CLASS_MAPPING
)
from .ai_gen_model import (
    is_ai_gen_model_available,
    load_ai_gen_model,
    predict_ai_generated,
    preprocess_for_ai_gen_detector,
    DEFAULT_AI_GEN_CLASS_MAPPING,
)
from .utils import generate_pdf_report


# ─────────────────────────────────────────────────────────────────────────────
# Helper factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_jpeg_file(width=100, height=100, color=(73, 109, 137), name="face.jpg"):
    """Creates a minimal valid JPEG SimpleUploadedFile."""
    buf = io.BytesIO()
    Image.new('RGB', (width, height), color=color).save(buf, format='JPEG')
    buf.seek(0)
    return SimpleUploadedFile(name, buf.getvalue(), content_type="image/jpeg")


def _make_bgr_array(fill=128, shape=(224, 224, 3)):
    """Creates a uniform BGR uint8 NumPy array."""
    return np.full(shape, fill, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Test Classes
# ─────────────────────────────────────────────────────────────────────────────

class PublicPagesTests(TestCase):
    """Basic smoke tests: public pages must return HTTP 200."""

    def setUp(self):
        self.client = Client()

    def test_public_pages_status_code(self):
        """Verify all public pages render with HTTP 200."""
        for url_name in ['home', 'about', 'how_it_works', 'contact', 'login', 'register']:
            response = self.client.get(reverse(url_name))
            self.assertEqual(
                response.status_code, 200,
                f"Public page '{url_name}' returned {response.status_code} instead of 200."
            )

    def test_protected_pages_redirect_unauthenticated(self):
        """Verify protected pages redirect unauthenticated users to login."""
        for url_name in ['dashboard', 'detect', 'history', 'profile']:
            response = self.client.get(reverse(url_name))
            self.assertEqual(
                response.status_code, 302,
                f"Protected page '{url_name}' should redirect but returned {response.status_code}."
            )


class AuthenticatedViewTests(TestCase):
    """Tests requiring a logged-in user."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='teststudent',
            password='Password123!',
            email='teststudent@example.com',
            first_name='Test',
            last_name='Student'
        )
        self.other_user = User.objects.create_user(
            username='otherstudent',
            password='Password123!',
            email='other@example.com'
        )

    def test_authenticated_dashboard_loads(self):
        """Dashboard loads with HTTP 200 for authenticated user."""
        self.client.login(username='teststudent', password='Password123!')
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Welcome, Test!')

    def test_user_history_privacy_isolation(self):
        """User A cannot view User B's detection result — should return 404."""
        record_b = DetectionResult.objects.create(
            user=self.other_user,
            file_name="private_other.jpg",
            file_type="IMAGE",
            file_size_mb=0.5,
            result="REAL",
            processing_status="COMPLETED"
        )
        self.client.login(username='teststudent', password='Password123!')
        response = self.client.get(reverse('detection_result', kwargs={'pk': record_b.pk}))
        self.assertEqual(response.status_code, 404)

    def test_history_filter_ai_generated(self):
        """History view ai_generated filter returns only AI_GENERATED records."""
        # Create one AI_GENERATED record and one REAL record
        DetectionResult.objects.create(
            user=self.user, file_name="ai_img.jpg", file_type="IMAGE",
            file_size_mb=0.5, result="AI_GENERATED", processing_status="COMPLETED"
        )
        DetectionResult.objects.create(
            user=self.user, file_name="real_img.jpg", file_type="IMAGE",
            file_size_mb=0.5, result="REAL", processing_status="COMPLETED"
        )
        self.client.login(username='teststudent', password='Password123!')
        response = self.client.get(reverse('history') + '?filter=ai_generated')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'ai_img.jpg')
        self.assertNotContains(response, 'real_img.jpg')


class FileValidationTests(TestCase):
    """Tests for the upload validator."""

    def test_rejects_disallowed_extension(self):
        """Executable files must be rejected."""
        f = SimpleUploadedFile("malicious.exe", b"binary", content_type="application/octet-stream")
        is_valid, err, _ = validate_uploaded_media(f)
        self.assertFalse(is_valid)
        self.assertIn("Unsupported file format", err)

    def test_rejects_empty_file(self):
        """Empty files must be rejected."""
        f = SimpleUploadedFile("empty.jpg", b"", content_type="image/jpeg")
        is_valid, err, _ = validate_uploaded_media(f)
        self.assertFalse(is_valid)
        self.assertIn("empty", err.lower())

    def test_accepts_valid_jpeg(self):
        """A valid JPEG image must be accepted as IMAGE type."""
        f = _make_jpeg_file()
        is_valid, err, media_type = validate_uploaded_media(f)
        self.assertTrue(is_valid, f"Valid JPEG was rejected: {err}")
        self.assertEqual(media_type, "IMAGE")


class DeepfakeModelTests(TestCase):
    """Tests for the deepfake detector (ml_model.py)."""

    def test_model_discovery_finds_onnx(self):
        """Classifier discovery should find deepfake_detector_v1.onnx."""
        info = find_deepfake_classifier()
        self.assertTrue(info['found'], "Trained model should be in model/ directory.")
        self.assertEqual(info['format'], 'ONNX')
        self.assertEqual(info['filename'], 'deepfake_detector_v1.onnx')

    def test_discovery_excludes_dummy_model(self):
        """The dummy deepfake_model.onnx must NOT be discovered as a classifier."""
        info = find_deepfake_classifier()
        # The only file returned must be the genuine model
        self.assertNotEqual(info.get('filename'), 'deepfake_model.onnx',
                            "Dummy model should not be returned by find_deepfake_classifier().")

    def test_onnx_session_loads(self):
        """ONNX Runtime session must load successfully with correct tensor names."""
        session, model_info = load_deepfake_model()
        self.assertIsNotNone(session, "ONNX session should load successfully.")
        self.assertTrue(model_info['available'])
        self.assertIn('ONNX Runtime', model_info['runtime'])
        self.assertEqual(model_info['input_name'], 'pixel_values')
        self.assertEqual(model_info['output_name'], 'logits')

    def test_siglip_preprocessing_shape_and_range(self):
        """SigLIP preprocessing must produce (1, 3, 224, 224) float32 in [-1, 1]."""
        dummy_bgr = _make_bgr_array(fill=204)  # 204 / 255.0 = 0.8 → (0.8 - 0.5) / 0.5 = 0.6
        tensor = preprocess_for_siglip(dummy_bgr)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 0.6, places=2)
        self.assertTrue(np.all(tensor >= -1.05) and np.all(tensor <= 1.05))

    def test_class_mapping_integrity(self):
        """Class index 0 must be FAKE and class index 1 must be REAL."""
        self.assertEqual(CLASS_MAPPING[0], 'FAKE')
        self.assertEqual(CLASS_MAPPING[1], 'REAL')

    def test_prediction_returns_valid_three_state_verdict(self):
        """Prediction must return REAL, FAKE, or UNCERTAIN with valid probability sums."""
        crop1 = _make_bgr_array(fill=100)
        crop2 = _make_bgr_array(fill=220)

        pred1 = predict_deepfake([crop1], media_type="IMAGE")
        pred2 = predict_deepfake([crop2], media_type="IMAGE")

        self.assertIn(pred1['prediction'], ['REAL', 'FAKE', 'UNCERTAIN'])
        self.assertTrue(0.0 <= pred1['confidence'] <= 100.0)
        self.assertTrue(0.0 <= pred1['real_probability'] <= 1.0)
        self.assertTrue(0.0 <= pred1['fake_probability'] <= 1.0)
        self.assertAlmostEqual(pred1['real_probability'] + pred1['fake_probability'], 1.0, places=3)
        self.assertEqual(pred1['faces_count'], 1)

    def test_prediction_is_input_specific(self):
        """Different inputs must produce different raw probabilities."""
        crop1 = _make_bgr_array(fill=50)
        crop2 = _make_bgr_array(fill=210)
        pred1 = predict_deepfake([crop1], media_type="IMAGE")
        pred2 = predict_deepfake([crop2], media_type="IMAGE")
        self.assertNotEqual(
            pred1['fake_probability'], pred2['fake_probability'],
            "Inputs with very different pixel values must produce different model scores."
        )

    def test_empty_face_crops_returns_uncertain(self):
        """Calling predict_deepfake with no face crops must return UNCERTAIN / NO_FACE."""
        result = predict_deepfake([], media_type="IMAGE")
        self.assertEqual(result['prediction'], 'UNCERTAIN')
        self.assertEqual(result['analysis_status'], 'NO_FACE')
        self.assertEqual(result['faces_count'], 0)


class AIGenModelTests(TestCase):
    """Tests for the AI-generated image detector (ai_gen_model.py)."""

    def test_default_class_mapping(self):
        """Default class mapping must assign index 0 to AI_GENERATED and 1 to REAL."""
        self.assertEqual(DEFAULT_AI_GEN_CLASS_MAPPING[0], 'AI_GENERATED')
        self.assertEqual(DEFAULT_AI_GEN_CLASS_MAPPING[1], 'REAL')

    def test_ai_gen_preprocessor_shape_and_range(self):
        """AI-gen preprocessor must produce (1, 3, 224, 224) float32 with ViT normalization (mean=0.5, std=0.5)."""
        dummy_bgr = _make_bgr_array(fill=128)
        tensor = preprocess_for_ai_gen_detector(dummy_bgr)

        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)

        # pixel=128: scaled=0.502, R-channel: (0.50196 - 0.5) / 0.5 ≈ 0.00392
        r_val = float(tensor[0, 0, 0, 0])
        expected_r = (128.0 / 255.0 - 0.5) / 0.5
        self.assertAlmostEqual(r_val, expected_r, places=2)

    def test_ai_gen_preprocessor_rejects_empty(self):
        """AI-gen preprocessor must raise ValueError on empty input."""
        with self.assertRaises(ValueError):
            preprocess_for_ai_gen_detector(np.array([]))

    def test_ai_gen_model_availability_check(self):
        """is_ai_gen_model_available() must return a boolean without crashing."""
        result = is_ai_gen_model_available()
        self.assertIsInstance(result, bool)

    def test_predict_ai_generated_unavailable_graceful(self):
        """When AI-gen model is not installed, predict_ai_generated must return UNAVAILABLE gracefully."""
        if is_ai_gen_model_available():
            self.skipTest("AI-gen model is installed — skipping unavailable-path test.")
        dummy_bgr = _make_bgr_array(fill=100)
        result = predict_ai_generated(dummy_bgr)
        self.assertFalse(result['available'])
        self.assertEqual(result['prediction'], 'UNAVAILABLE')
        self.assertIsNotNone(result['error'])
        self.assertEqual(result['ai_gen_probability'], 0.0)

    def test_predict_ai_generated_when_available(self):
        """When AI-gen model is installed, predict_ai_generated must return valid predictions."""
        if not is_ai_gen_model_available():
            self.skipTest("AI-gen model not installed — download it to run this test.")
        dummy_bgr = _make_bgr_array(fill=100)
        result = predict_ai_generated(dummy_bgr)
        self.assertTrue(result['available'])
        self.assertIn(result['prediction'], ['AI_GENERATED', 'REAL', 'UNCERTAIN'])
        self.assertTrue(0.0 <= result['ai_gen_probability'] <= 1.0)
        self.assertTrue(0.0 <= result['real_probability'] <= 1.0)
        self.assertTrue(0.0 <= result['confidence'] <= 100.0)

    def test_predict_ai_generated_missing_file(self):
        """predict_ai_generated must return UNAVAILABLE when file path does not exist."""
        result = predict_ai_generated("/nonexistent/path/image.jpg")
        self.assertFalse(result['available'])
        self.assertEqual(result['prediction'], 'UNAVAILABLE')
        self.assertIsNotNone(result['error'])


class CombinedVerdictLogicTests(TestCase):
    """Unit tests for the four-state combined verdict logic (without Django views)."""

    def _combined_verdict(self, ai_gen_pred, ai_gen_available, deepfake_pred):
        """
        Replicates the combined verdict logic from detect_upload_view.
        Returns the combined verdict string.
        """
        if ai_gen_available and ai_gen_pred == 'AI_GENERATED':
            return 'AI_GENERATED'
        elif deepfake_pred == 'FAKE':
            return 'FAKE'
        elif deepfake_pred == 'REAL' and (not ai_gen_available or ai_gen_pred in ('REAL', 'UNCERTAIN')):
            return 'REAL'
        else:
            return 'UNCERTAIN'

    def test_ai_gen_wins_over_deepfake_real(self):
        """AI_GENERATED from AI-gen detector overrides REAL from deepfake detector."""
        self.assertEqual(self._combined_verdict('AI_GENERATED', True, 'REAL'), 'AI_GENERATED')

    def test_ai_gen_wins_over_deepfake_fake(self):
        """AI_GENERATED from AI-gen detector overrides FAKE from deepfake detector."""
        self.assertEqual(self._combined_verdict('AI_GENERATED', True, 'FAKE'), 'AI_GENERATED')

    def test_fake_when_deepfake_fires_and_ai_gen_real(self):
        """FAKE verdict when deepfake detector fires and AI-gen says REAL."""
        self.assertEqual(self._combined_verdict('REAL', True, 'FAKE'), 'FAKE')

    def test_fake_when_deepfake_fires_and_ai_gen_unavailable(self):
        """FAKE verdict when deepfake fires and AI-gen model is unavailable."""
        self.assertEqual(self._combined_verdict('UNAVAILABLE', False, 'FAKE'), 'FAKE')

    def test_real_when_both_agree(self):
        """REAL verdict when both models agree."""
        self.assertEqual(self._combined_verdict('REAL', True, 'REAL'), 'REAL')

    def test_real_when_ai_gen_unavailable_and_deepfake_real(self):
        """REAL verdict when AI-gen unavailable and deepfake says REAL."""
        self.assertEqual(self._combined_verdict('UNAVAILABLE', False, 'REAL'), 'REAL')

    def test_uncertain_when_ai_gen_uncertain_and_deepfake_uncertain(self):
        """UNCERTAIN when both models are uncertain."""
        self.assertEqual(self._combined_verdict('UNCERTAIN', True, 'UNCERTAIN'), 'UNCERTAIN')

    def test_uncertain_when_ai_gen_uncertain_and_deepfake_real(self):
        """REAL when AI-gen uncertain and deepfake says REAL (uncertain does not override)."""
        self.assertEqual(self._combined_verdict('UNCERTAIN', True, 'REAL'), 'REAL')


class DatabaseModelTests(TestCase):
    """Tests for Django ORM models including the new AI_GENERATED choice and fields."""

    def setUp(self):
        self.user = User.objects.create_user(username='dbtest', password='Test123!')

    def test_create_ai_generated_result(self):
        """DetectionResult.result='AI_GENERATED' must save and retrieve correctly."""
        record = DetectionResult.objects.create(
            user=self.user,
            file_name="synthetic_portrait.jpg",
            file_type="IMAGE",
            file_size_mb=0.8,
            result="AI_GENERATED",
            confidence=82.5,
            ai_generated_probability=0.825,
            ai_gen_model_name="onnx-community/ai-image-detect-distilled-ONNX",
            ai_gen_model_available=True,
            processing_status="COMPLETED",
            face_detected=True,
            faces_count=1,
        )
        fetched = DetectionResult.objects.get(pk=record.pk)
        self.assertEqual(fetched.result, 'AI_GENERATED')
        self.assertTrue(fetched.is_ai_generated)
        self.assertFalse(fetched.is_fake)
        self.assertFalse(fetched.is_real)
        self.assertEqual(fetched.badge_class, 'bg-purple')
        self.assertAlmostEqual(fetched.ai_generated_probability, 0.825, places=3)
        self.assertTrue(fetched.ai_gen_model_available)

    def test_result_choices_include_ai_generated(self):
        """RESULT_CHOICES must include AI_GENERATED."""
        choices_keys = [c[0] for c in DetectionResult.RESULT_CHOICES]
        self.assertIn('AI_GENERATED', choices_keys)
        self.assertIn('REAL', choices_keys)
        self.assertIn('FAKE', choices_keys)
        self.assertIn('UNCERTAIN', choices_keys)

    def test_face_detection_item_relational_storage(self):
        """FaceDetectionItem records must link correctly to DetectionResult."""
        record = DetectionResult.objects.create(
            user=self.user, file_name="group.jpg", file_type="IMAGE",
            file_size_mb=0.8, result="REAL", confidence=88.5,
            real_probability=0.885, fake_probability=0.115,
            processing_status="COMPLETED", face_detected=True, faces_count=2
        )
        f1 = FaceDetectionItem.objects.create(
            detection_result=record, face_number=1,
            bounding_box_x=50, bounding_box_y=50, bounding_box_w=120, bounding_box_h=140,
            label="REAL", confidence=89.2, real_score=0.892, fake_score=0.108
        )
        f2 = FaceDetectionItem.objects.create(
            detection_result=record, face_number=2,
            bounding_box_x=220, bounding_box_y=60, bounding_box_w=110, bounding_box_h=130,
            label="REAL", confidence=87.8, real_score=0.878, fake_score=0.122
        )
        self.assertEqual(record.face_items.count(), 2)
        self.assertEqual(f1.detection_result, record)

    def test_no_face_detection_result_properties(self):
        """NO_FACE result must be exposed by is_uncertain property."""
        record = DetectionResult.objects.create(
            user=self.user, file_name="blank.jpg", file_type="IMAGE",
            file_size_mb=0.1, result="UNCERTAIN", processing_status="NO_FACE", face_detected=False
        )
        self.assertTrue(record.is_uncertain)
        self.assertFalse(record.is_ai_generated)


class PDFReportTests(TestCase):
    """Tests for PDF report generation."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='pdftest', password='Test123!',
            email='pdftest@example.com', first_name='PDF', last_name='User'
        )

    def test_pdf_report_generated_for_real_result(self):
        """PDF report must download with correct content-type for a REAL result."""
        record = DetectionResult.objects.create(
            user=self.user, file_name="sample_verify.jpg", file_type="IMAGE",
            file_size_mb=1.2, result="REAL", confidence=95.5,
            real_probability=0.955, fake_probability=0.045,
            processing_status="COMPLETED", face_detected=True
        )
        self.client.login(username='pdftest', password='Test123!')
        response = self.client.get(reverse('report', kwargs={'pk': record.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertIn('Deepfake_Report', response['Content-Disposition'])

    def test_pdf_report_generated_for_ai_generated_result(self):
        """PDF report must download correctly for an AI_GENERATED verdict."""
        record = DetectionResult.objects.create(
            user=self.user, file_name="synthetic.jpg", file_type="IMAGE",
            file_size_mb=0.9, result="AI_GENERATED", confidence=84.0,
            ai_generated_probability=0.840, ai_gen_model_available=True,
            ai_gen_model_name="onnx-community ai-image-detect-distilled-ONNX",
            processing_status="COMPLETED", face_detected=True, faces_count=1,
            forensic_details={
                'ai_gen_result': {
                    'prediction': 'AI_GENERATED',
                    'ai_gen_probability': 0.840,
                    'real_probability': 0.160,
                    'confidence': 84.0,
                    'engine': 'onnx-community ai-image-detect-distilled-ONNX',
                    'available': True,
                    'error': None,
                }
            }
        )
        self.client.login(username='pdftest', password='Test123!')
        response = self.client.get(reverse('report', kwargs={'pk': record.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')

    def test_analysis_report_tracking(self):
        """AnalysisReport record must be created upon PDF download."""
        record = DetectionResult.objects.create(
            user=self.user, file_name="tracked.jpg", file_type="IMAGE",
            file_size_mb=0.5, result="REAL", confidence=90.0,
            processing_status="COMPLETED", face_detected=True
        )
        self.client.login(username='pdftest', password='Test123!')
        self.client.get(reverse('report', kwargs={'pk': record.pk}))
        reports = AnalysisReport.objects.filter(detection_result=record)
        self.assertTrue(reports.exists())
        self.assertEqual(reports.first().generated_by, self.user)
