import io
import os
from django.conf import settings
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def generate_pdf_report(detection_result, report_id=None):
    """
    Generates a professional forensic PDF report for a given DetectionResult object
    and returns a downloadable HttpResponse.
    Handles single face, multiple faces, trained model predictions, and model unavailable states.
    Now includes dual-model signal summary (AI-gen + Deepfake detectors).
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1e3a8a'),
        alignment=TA_CENTER
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b'),
        alignment=TA_CENTER
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=5
    )

    cell_label = ParagraphStyle(
        'CellLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#334155')
    )

    cell_value = ParagraphStyle(
        'CellValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0f172a')
    )

    is_model_unavailable = detection_result.processing_status == 'MODEL_UNAVAILABLE'
    is_no_face = detection_result.processing_status == 'NO_FACE'
    is_ai_generated = detection_result.result == 'AI_GENERATED'
    is_inconclusive = detection_result.result == 'UNCERTAIN' or is_model_unavailable or is_no_face

    # Result Badge Style
    if is_ai_generated:
        result_color = colors.HexColor('#7c3aed')
        result_bg = colors.HexColor('#f5f3ff')
        verdict_text = "LIKELY AI-GENERATED IMAGE"
    elif detection_result.result == 'REAL' and not is_inconclusive:
        result_color = colors.HexColor('#16a34a')
        result_bg = colors.HexColor('#f0fdf4')
        verdict_text = "AUTHENTIC / REAL MEDIA"
    elif detection_result.result == 'FAKE' and not is_inconclusive:
        result_color = colors.HexColor('#dc2626')
        result_bg = colors.HexColor('#fef2f2')
        verdict_text = "DEEPFAKE / MANIPULATION DETECTED"
    elif is_no_face:
        result_color = colors.HexColor('#d97706')
        result_bg = colors.HexColor('#fffbeb')
        verdict_text = "NO FACE DETECTED"
    elif is_model_unavailable:
        result_color = colors.HexColor('#d97706')
        result_bg = colors.HexColor('#fffbeb')
        verdict_text = "ANALYSIS INCONCLUSIVE<br/><font size=9 color='#b45309'>Model Unavailable</font>"
    else:
        result_color = colors.HexColor('#d97706')
        result_bg = colors.HexColor('#fffbeb')
        verdict_text = "ANALYSIS INCONCLUSIVE<br/><font size=9 color='#b45309'>Borderline Indicators</font>"

    result_badge_style = ParagraphStyle(
        'ResultBadge',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=result_color,
        alignment=TA_CENTER
    )

    confidence_style = ParagraphStyle(
        'ConfidenceScore',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=result_color,
        alignment=TA_CENTER
    )

    # Header section
    story.append(Paragraph("DEEPFAKE DETECTION SYSTEM", title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("FORENSIC MEDIA ANALYSIS REPORT &amp; VERIFICATION CERTIFICATE", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceAfter=12))

    # Certificate / Summary Card
    if is_inconclusive or detection_result.confidence == 0.0:
        confidence_cell_content = (
            "<b>CONFIDENCE LEVEL:</b><br/>"
            "<font size=16>N/A</font><br/>"
            "<font size=7.5 color='#64748b'>Uncertainty threshold triggered or model absent.</font>"
        )
    elif is_ai_generated:
        ai_gen_p = detection_result.ai_generated_probability
        confidence_cell_content = (
            f"<b>CONFIDENCE LEVEL:</b><br/>"
            f"<font size=16>{detection_result.confidence:.1f}%</font><br/>"
            f"<font size=7.5 color='#64748b'>AI-Gen: {ai_gen_p*100:.1f}% | Real: {(1-ai_gen_p)*100:.1f}%</font>"
        )
    else:
        confidence_cell_content = (
            f"<b>CONFIDENCE LEVEL:</b><br/>"
            f"<font size=16>{detection_result.confidence:.1f}%</font><br/>"
            f"<font size=7.5 color='#64748b'>Real: {detection_result.real_probability*100:.1f}% | Fake: {detection_result.fake_probability*100:.1f}%</font>"
        )

    summary_table_data = [
        [
            Paragraph(f"<b>ANALYSIS VERDICT:</b><br/>{verdict_text}", result_badge_style),
            Paragraph(confidence_cell_content, confidence_style)
        ]
    ]
    summary_table = Table(summary_table_data, colWidths=[3.8 * inch, 3.8 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), result_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, result_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # Metadata Section
    story.append(Paragraph("FILE &amp; INSPECTION METADATA", section_heading))
    formatted_date = detection_result.detected_at.strftime("%B %d, %Y at %H:%M:%S IST")
    actual_report_id = report_id or f"DFS-{detection_result.id:06d}"

    meta_data = [
        [Paragraph("Report ID:", cell_label), Paragraph(actual_report_id, cell_value),
         Paragraph("Analysis Date:", cell_label), Paragraph(formatted_date, cell_value)],
        [Paragraph("Submitted By:", cell_label), Paragraph(f"{detection_result.user.get_full_name() or detection_result.user.username} (@{detection_result.user.username})", cell_value),
         Paragraph("User Email:", cell_label), Paragraph(detection_result.user.email or "N/A", cell_value)],
        [Paragraph("File Name:", cell_label), Paragraph(detection_result.file_name, cell_value),
         Paragraph("Media Type:", cell_label), Paragraph(f"{detection_result.get_file_type_display()} (.{detection_result.file_extension})", cell_value)],
        [Paragraph("File Size:", cell_label), Paragraph(f"{detection_result.file_size_mb:.2f} MB", cell_value),
         Paragraph("Processing Status:", cell_label), Paragraph(detection_result.get_processing_status_display(), cell_value)],
        [Paragraph("Face Detected:", cell_label), Paragraph(f"Yes ({detection_result.faces_count} located)" if detection_result.face_detected else "No", cell_value),
         Paragraph("Frames / Samples:", cell_label), Paragraph(f"{detection_result.frames_analyzed} frames analyzed", cell_value)],
        [Paragraph("Engine Model:", cell_label), Paragraph(detection_result.model_name, cell_value),
         Paragraph("Calibration:", cell_label), Paragraph(detection_result.calibration_status, cell_value)],
    ]

    meta_table = Table(meta_data, colWidths=[1.4 * inch, 2.4 * inch, 1.4 * inch, 2.4 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Face Preview
    if detection_result.extracted_face:
        face_img_path = os.path.join(settings.MEDIA_ROOT, detection_result.extracted_face.name)
        if os.path.exists(face_img_path):
            story.append(Paragraph("DETECTED FACIAL BIOMETRIC REGION", section_heading))
            try:
                face_rl_img = RLImage(face_img_path, width=2.2 * inch, height=2.2 * inch)
                face_rl_img.hAlign = 'CENTER'
                story.append(face_rl_img)
                story.append(Spacer(1, 10))
            except Exception as e:
                print(f"Error adding image to PDF: {e}")

    # Dual-Model Signal Summary
    story.append(Paragraph("DUAL-MODEL DETECTION SIGNAL SUMMARY", section_heading))
    forensic = detection_result.forensic_details or {}
    ai_gen_data = forensic.get('ai_gen_result', {})
    ai_gen_pred = ai_gen_data.get('prediction', 'N/A')
    ai_gen_prob = ai_gen_data.get('ai_gen_probability', 0.0)
    ai_gen_conf = ai_gen_data.get('confidence', 0.0)
    ai_gen_avail = ai_gen_data.get('available', False)
    deepfake_label = 'N/A (AI-Generated)' if is_ai_generated else detection_result.result

    dual_model_data = [
        [Paragraph("Model", cell_label), Paragraph("Purpose", cell_label),
         Paragraph("Verdict", cell_label), Paragraph("Probability", cell_label), Paragraph("Confidence", cell_label)],
        [
            Paragraph("AI-Image Detect (onnx-community)", cell_value),
            Paragraph("Whole-image AI generation detection", cell_value),
            Paragraph(ai_gen_pred if ai_gen_avail else "Not Installed", cell_value),
            Paragraph(f"{ai_gen_prob:.3f}" if ai_gen_avail else "-", cell_value),
            Paragraph(f"{ai_gen_conf:.1f}%" if ai_gen_avail else "-", cell_value),
        ],
        [
            Paragraph("prithivML SigLIP Deepfake", cell_value),
            Paragraph("Face-crop deepfake/manipulation detection", cell_value),
            Paragraph(deepfake_label, cell_value),
            Paragraph(f"{detection_result.fake_probability:.3f}", cell_value),
            Paragraph(f"{detection_result.confidence:.1f}%" if not is_inconclusive else "N/A", cell_value),
        ],
    ]
    dual_table = Table(dual_model_data, colWidths=[1.8 * inch, 2.0 * inch, 1.4 * inch, 1.0 * inch, 1.4 * inch])
    dual_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0f2fe')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(dual_table)
    story.append(Spacer(1, 12))

    # Relational Face-Level & Frame-Level Results
    face_items = list(detection_result.face_items.all())
    if face_items:
        story.append(Paragraph(f"FACE &amp; FRAME ANALYSIS BREAKDOWN ({len(face_items)} REGIONS ANALYZED)", section_heading))
        face_rows = [
            [Paragraph("Face #", cell_label), Paragraph("Bounding Box", cell_label), Paragraph("Frame / Time", cell_label), Paragraph("Prediction", cell_label), Paragraph("Score", cell_label), Paragraph("Texture Std", cell_label)]
        ]
        for f in face_items:
            bbox_str = f"({f.bounding_box_x},{f.bounding_box_y},{f.bounding_box_w},{f.bounding_box_h})"
            time_str = f"Frame {f.frame_number} ({f.frame_timestamp:.1f}s)" if f.frame_timestamp > 0 else f"Frame {f.frame_number}"
            conf_str = f"{f.confidence:.1f}%" if f.confidence > 0 else "N/A"
            face_rows.append([
                Paragraph(f"Face #{f.face_number}", cell_value),
                Paragraph(bbox_str, cell_value),
                Paragraph(time_str, cell_value),
                Paragraph(f.label, cell_value),
                Paragraph(conf_str, cell_value),
                Paragraph(f"{f.laplacian_variance:.1f}", cell_value),
            ])

        faces_table = Table(face_rows, colWidths=[1.1 * inch, 1.6 * inch, 1.5 * inch, 1.2 * inch, 1.0 * inch, 1.2 * inch])
        faces_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0f2fe')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffffff')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(faces_table)
        story.append(Spacer(1, 12))

    # Forensic Details Breakdown
    story.append(Paragraph("SUPPORTING FORENSIC TELEMETRY", section_heading))
    forensic_rows = [
        [Paragraph("Forensic Metric", cell_label), Paragraph("Value", cell_label), Paragraph("Interpretation", cell_label)],
        [
            Paragraph("Laplacian Texture Variance", cell_value),
            Paragraph(str(forensic.get('laplacian_sharpness_var', 'N/A')), cell_value),
            Paragraph("Spatial edge crispness and optical texture variance.", cell_value)
        ],
        [
            Paragraph("Local Patch Texture Std", cell_value),
            Paragraph(str(forensic.get('local_patch_texture_std', 'N/A')), cell_value),
            Paragraph("Micro-texture consistency across non-overlapping patches.", cell_value)
        ],
        [
            Paragraph("Sensor Noise Residual", cell_value),
            Paragraph(str(forensic.get('noise_residual_estimate', 'N/A')), cell_value),
            Paragraph("High-frequency camera sensor residual noise estimate.", cell_value)
        ],
        [
            Paragraph("High-Frequency Spectral Energy", cell_value),
            Paragraph(str(forensic.get('high_frequency_energy', 'N/A')), cell_value),
            Paragraph("2D FFT inspection for synthetic frequency cutoffs.", cell_value)
        ],
        [
            Paragraph("Analysis Engine", cell_value),
            Paragraph(str(forensic.get('engine', detection_result.model_name)), cell_value),
            Paragraph("Inference runtime pipeline utilized for this scan.", cell_value)
        ],
    ]

    forensic_table = Table(forensic_rows, colWidths=[2.2 * inch, 1.6 * inch, 3.8 * inch])
    forensic_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0f2fe')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(forensic_table)
    story.append(Spacer(1, 12))

    # Disclaimer & Signature Footer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#cbd5e1'), spaceAfter=6))
    video_note = (
        " Video analysis was conducted frame-by-frame on sampled keyframes; "
        "temporal consistency is an aggregate estimation and does not replace dedicated spatiotemporal video models."
    ) if detection_result.file_type == 'VIDEO' else ""
    ai_gen_note = (
        f" AI-gen detector result: {ai_gen_pred} ({ai_gen_prob:.1%} AI-gen probability, Apache-2.0 model)."
    ) if ai_gen_avail else " AI-generated image detector was not installed during this scan."
    disclaimer_text = (
        "<b>Important Limitations &amp; Transparency Notice:</b> This report is generated automatically by the "
        "Deepfake Detection System. This system uses two independent AI models: "
        "(1) the pretrained SigLIP vision transformer (prithivML-deepfake-detector-v1) for face-level deepfake detection, "
        "and (2) a distilled ViT AI-image classifier (onnx-community, Apache-2.0) for whole-image AI generation detection. "
        "Model prediction scores indicate statistical algorithmic confidence based on learned visual artifacts and "
        "are NOT a certified guarantee or infallible proof of authenticity. "
        "Modern high-quality AI-generated images and deepfakes may evade detection by both models."
        + ai_gen_note + video_note
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor('#64748b'),
        alignment=TA_LEFT
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    buffer.seek(0)

    filename = f"Deepfake_Report_{detection_result.id:06d}.pdf"
    response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response
