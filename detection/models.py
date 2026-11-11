import os
from django.db import models
from django.contrib.auth.models import User


class DetectionResult(models.Model):
    """
    Stores the complete metadata and AI forensic results for an uploaded image or video.
    """
    FILE_TYPE_CHOICES = [
        ('IMAGE', 'Image'),
        ('VIDEO', 'Video'),
    ]

    RESULT_CHOICES = [
        ('REAL', 'Real / Authentic'),
        ('FAKE', 'Fake / Deepfake Detected'),
        ('AI_GENERATED', 'AI-Generated Image'),
        ('UNCERTAIN', 'Inconclusive / Low Quality'),
    ]

    STATUS_CHOICES = [
        ('PENDING', 'Pending Analysis'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Analysis Completed'),
        ('NO_FACE', 'No Face Detected'),
        ('FAILED', 'Analysis Failed'),
        ('MODEL_UNAVAILABLE', 'Model Unavailable'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='detection_results',
        help_text="User who uploaded and requested the detection."
    )
    file = models.FileField(
        upload_to='uploads/%Y/%m/%d/',
        help_text="Uploaded media file (Image or Video)."
    )
    file_name = models.CharField(max_length=255, blank=True)
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    file_size_mb = models.FloatField(default=0.0)
    
    # AI Prediction Metrics
    result = models.CharField(max_length=20, choices=RESULT_CHOICES, default='UNCERTAIN')
    confidence = models.FloatField(default=0.0, help_text="Confidence percentage (0 - 100%)")
    fake_probability = models.FloatField(default=0.0, help_text="Probability of media being Fake / Deepfake (0.0 - 1.0)")
    real_probability = models.FloatField(default=0.0, help_text="Probability of media being Real (0.0 - 1.0)")

    # AI-Generated Image Detection (separate from deepfake face-swap detection)
    ai_generated_probability = models.FloatField(
        default=0.0,
        help_text="Probability of image being fully AI-generated (0.0 - 1.0)"
    )
    ai_gen_model_name = models.CharField(
        max_length=150,
        default='',
        blank=True,
        help_text="Name of the AI-generated image detection model used."
    )
    ai_gen_model_available = models.BooleanField(
        default=False,
        help_text="Whether the AI-gen detection model was loaded and run."
    )
    
    # Face Detection metadata
    face_detected = models.BooleanField(default=False)
    faces_count = models.IntegerField(default=0)
    extracted_face = models.ImageField(
        upload_to='processed/faces/%Y/%m/%d/',
        null=True,
        blank=True,
        help_text="Cropped face or annotated preview."
    )
    frames_analyzed = models.IntegerField(default=1)
    
    # Engine & Forensic Analysis
    model_name = models.CharField(max_length=100, default="CNN Deepfake Classifier v2.1")
    model_version = models.CharField(max_length=50, default="1.0.0", blank=True)
    calibration_status = models.CharField(max_length=50, default="Uncalibrated Sigmoid", blank=True)
    uncertainty_reason = models.CharField(max_length=255, blank=True, null=True)

    forensic_details = models.JSONField(
        default=dict,
        blank=True,
        help_text="Detailed forensic metrics like noise variance, frequency artifacts, blur ratio."
    )
    
    processing_status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='PENDING'
    )
    error_message = models.TextField(blank=True, null=True)
    detected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-detected_at']
        verbose_name = 'Detection Result'
        verbose_name_plural = 'Detection Results'

    def __str__(self):
        return f"{self.user.username} | {self.file_name} | {self.result} ({self.confidence:.1f}%)"

    @property
    def is_fake(self):
        return self.result == 'FAKE'

    @property
    def is_real(self):
        return self.result == 'REAL'

    @property
    def is_ai_generated(self):
        return self.result == 'AI_GENERATED'

    @property
    def is_uncertain(self):
        return self.result == 'UNCERTAIN' or self.processing_status in ['NO_FACE', 'MODEL_UNAVAILABLE']

    @property
    def badge_class(self):
        if self.result == 'REAL':
            return 'bg-success'
        elif self.result == 'FAKE':
            return 'bg-danger'
        elif self.result == 'AI_GENERATED':
            return 'bg-purple'
        elif self.processing_status in ['NO_FACE', 'MODEL_UNAVAILABLE']:
            return 'bg-warning text-dark'
        return 'bg-secondary'

    @property
    def file_extension(self):
        if self.file_name:
            _, ext = os.path.splitext(self.file_name)
            return ext.lower().replace('.', '')
        return ''


class FaceDetectionItem(models.Model):
    """
    Stores individual face and frame level analysis results.
    Allows multi-face and multi-frame inspection per detection result.
    """
    detection_result = models.ForeignKey(
        DetectionResult,
        on_delete=models.CASCADE,
        related_name='face_items'
    )
    face_number = models.PositiveIntegerField(default=1)
    bounding_box_x = models.IntegerField(default=0)
    bounding_box_y = models.IntegerField(default=0)
    bounding_box_w = models.IntegerField(default=0)
    bounding_box_h = models.IntegerField(default=0)
    face_crop = models.CharField(max_length=500, blank=True, null=True)

    label = models.CharField(max_length=20, default='UNCERTAIN')
    confidence = models.FloatField(default=0.0)
    real_score = models.FloatField(default=0.0)
    fake_score = models.FloatField(default=0.0)

    frame_number = models.PositiveIntegerField(default=1)
    frame_timestamp = models.FloatField(default=0.0)

    laplacian_variance = models.FloatField(default=0.0)
    frequency_energy = models.FloatField(default=0.0)
    noise_estimate = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['frame_number', 'face_number']
        verbose_name = 'Face Detection Item'
        verbose_name_plural = 'Face Detection Items'

    def __str__(self):
        return f"Result #{self.detection_result_id} - Face #{self.face_number} ({self.label})"


class AnalysisReport(models.Model):
    """
    Tracks generated PDF reports for an analysis result.
    """
    detection_result = models.ForeignKey(
        DetectionResult,
        on_delete=models.CASCADE,
        related_name='reports'
    )
    report_id = models.CharField(max_length=100, unique=True)
    generated_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='generated_reports'
    )
    generated_at = models.DateTimeField(auto_now_add=True)
    download_count = models.PositiveIntegerField(default=1)

    class Meta:
        ordering = ['-generated_at']
        verbose_name = 'Analysis Report'
        verbose_name_plural = 'Analysis Reports'

    def __str__(self):
        return f"Report {self.report_id} ({self.detection_result.file_name})"
