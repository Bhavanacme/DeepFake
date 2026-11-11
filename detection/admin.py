from django.contrib import admin
from django.utils.html import format_html
from .models import DetectionResult, FaceDetectionItem, AnalysisReport


class FaceDetectionItemInline(admin.TabularInline):
    model = FaceDetectionItem
    extra = 0
    readonly_fields = (
        'face_number', 'bounding_box_display', 'label_badge',
        'confidence_display', 'real_score', 'fake_score',
        'frame_number', 'frame_timestamp', 'laplacian_variance'
    )
    can_delete = False

    def bounding_box_display(self, obj):
        return f"({obj.bounding_box_x}, {obj.bounding_box_y}, {obj.bounding_box_w}, {obj.bounding_box_h})"
    bounding_box_display.short_description = "BBox (X, Y, W, H)"

    def label_badge(self, obj):
        color = "#16a34a" if obj.label == "REAL" else "#dc2626" if obj.label == "FAKE" else "#f59e0b"
        return format_html('<span style="background:{}; color:#fff; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:10px;">{}</span>', color, obj.label)
    label_badge.short_description = "Prediction"

    def confidence_display(self, obj):
        return f"{obj.confidence:.1f}%"
    confidence_display.short_description = "Conf %"


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'user',
        'file_name',
        'file_type_badge',
        'result_badge',
        'confidence_display',
        'processing_status_badge',
        'detected_at'
    )
    list_filter = ('result', 'file_type', 'processing_status', 'face_detected', 'detected_at')
    search_fields = ('file_name', 'user__username', 'user__email', 'result')
    readonly_fields = (
        'detected_at',
        'file_preview',
        'extracted_face_preview',
        'forensic_details_display'
    )
    ordering = ('-detected_at',)
    date_hierarchy = 'detected_at'
    inlines = [FaceDetectionItemInline]

    fieldsets = (
        ('General Information', {
            'fields': ('user', 'file', 'file_preview', 'file_name', 'file_type', 'file_size_mb')
        }),
        ('AI Detection & Classification', {
            'fields': ('result', 'confidence', 'fake_probability', 'real_probability', 'model_name', 'model_version', 'calibration_status', 'uncertainty_reason')
        }),
        ('AI-Generated Image Detection', {
            'fields': ('ai_generated_probability', 'ai_gen_model_name', 'ai_gen_model_available'),
            'classes': ('collapse',),
        }),
        ('Face Detection Telemetry', {
            'fields': ('face_detected', 'faces_count', 'frames_analyzed', 'extracted_face', 'extracted_face_preview')
        }),
        ('Forensic Analysis Details', {
            'fields': ('processing_status', 'error_message', 'forensic_details_display', 'detected_at')
        }),
    )

    def file_type_badge(self, obj):
        color = "#2563eb" if obj.file_type == "IMAGE" else "#7c3aed"
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;">{}</span>',
            color,
            obj.file_type
        )
    file_type_badge.short_description = 'Media Type'

    def result_badge(self, obj):
        if obj.result == 'REAL':
            bg = "#16a34a"
            text = "REAL"
        elif obj.result == 'FAKE':
            bg = "#dc2626"
            text = "DEEPFAKE"
        elif obj.result == 'AI_GENERATED':
            bg = "#7c3aed"
            text = "AI-GENERATED"
        else:
            bg = "#f59e0b"
            text = obj.result
        return format_html(
            '<span style="background-color: {}; color: white; padding: 4px 10px; border-radius: 9999px; font-weight: bold; font-size: 11px;">{}</span>',
            bg,
            text
        )
    result_badge.short_description = 'AI Verdict'

    def confidence_display(self, obj):
        color = "#16a34a" if obj.result == "REAL" else "#dc2626" if obj.result == "FAKE" else "#7c3aed" if obj.result == "AI_GENERATED" else "#f59e0b"
        return format_html(
            '<span style="font-weight: 600; color: {};">{:.1f}%</span>',
            color,
            obj.confidence
        )
    confidence_display.short_description = 'Confidence'

    def processing_status_badge(self, obj):
        status_colors = {
            'COMPLETED': '#10b981',
            'PROCESSING': '#3b82f6',
            'PENDING': '#6b7280',
            'NO_FACE': '#f59e0b',
            'FAILED': '#ef4444'
        }
        color = status_colors.get(obj.processing_status, '#6b7280')
        return format_html(
            '<span style="color: {}; font-weight: 600; font-size: 11px;">{}</span>',
            color,
            obj.get_processing_status_display()
        )
    processing_status_badge.short_description = 'Status'

    def file_preview(self, obj):
        if obj.file and obj.file_type == 'IMAGE':
            return format_html('<img src="{}" style="max-height: 180px; border-radius: 8px; border: 1px solid #ddd;" />', obj.file.url)
        elif obj.file and obj.file_type == 'VIDEO':
            return format_html('<video src="{}" controls style="max-height: 180px; border-radius: 8px;"></video>', obj.file.url)
        return "No file available"
    file_preview.short_description = 'Uploaded Media Preview'

    def extracted_face_preview(self, obj):
        if obj.extracted_face:
            return format_html('<img src="{}" style="max-height: 180px; border-radius: 8px; border: 1px solid #ddd;" />', obj.extracted_face.url)
        return "No face crop extracted"
    extracted_face_preview.short_description = 'Detected Face Preview'

    def forensic_details_display(self, obj):
        if not obj.forensic_details:
            return "No forensic telemetry recorded."
        html = "<table style='border-collapse:collapse; width:100%; font-size:12px;'>"
        for k, v in obj.forensic_details.items():
            if k == 'faces_analysis':
                continue
            html += f"<tr><td style='border:1px solid #eee; padding:4px 8px; font-weight:bold; width:220px;'>{k}</td><td style='border:1px solid #eee; padding:4px 8px;'>{v}</td></tr>"
        html += "</table>"
        return format_html(html)
    forensic_details_display.short_description = 'Forensic Signals'


@admin.register(FaceDetectionItem)
class FaceDetectionItemAdmin(admin.ModelAdmin):
    list_display = ('id', 'detection_result', 'face_number', 'label', 'confidence', 'frame_number', 'frame_timestamp', 'created_at')
    list_filter = ('label', 'created_at')
    search_fields = ('detection_result__file_name', 'label')


@admin.register(AnalysisReport)
class AnalysisReportAdmin(admin.ModelAdmin):
    list_display = ('report_id', 'detection_result', 'generated_by', 'generated_at', 'download_count')
    list_filter = ('generated_at',)
    search_fields = ('report_id', 'detection_result__file_name', 'generated_by__username')
