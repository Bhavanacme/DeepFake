import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import Http404
from django.core.paginator import Paginator
from django.db.models import Count, Q, Sum
from django.conf import settings
from django.utils import timezone
from django.contrib.auth.models import User
from django.contrib.admin.views.decorators import staff_member_required

from .models import DetectionResult, FaceDetectionItem, AnalysisReport
from .forms import MediaUploadForm
from .preprocessing import validate_uploaded_media, process_image, process_video
from .ml_model import predict_deepfake
from .ai_gen_model import predict_ai_generated
from .utils import generate_pdf_report


def home_view(request):
    """
    Public landing page introducing the AI Deepfake Detection System.
    Includes direct media upload card and recent inspection telemetry.
    """
    total_detections_platform = DetectionResult.objects.count()
    real_platform = DetectionResult.objects.filter(result='REAL').count()
    fake_platform = DetectionResult.objects.filter(result='FAKE').count()

    upload_form = MediaUploadForm()

    if request.user.is_authenticated:
        recent_history = DetectionResult.objects.filter(user=request.user)[:5]
    else:
        recent_history = []

    context = {
        'page_title': 'Deepfake Detector | AI-Powered Media Verification',
        'total_detections_platform': total_detections_platform,
        'real_platform': real_platform,
        'fake_platform': fake_platform,
        'upload_form': upload_form,
        'recent_history': recent_history,
    }
    return render(request, 'detection/home.html', context)


def contact_view(request):
    """
    Contact & Inquiries page for the Deepfake Detection System.
    """
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip()
        message = request.POST.get('message', '').strip()
        if name and email and message:
            messages.success(request, f"Thank you, {name}! Your message has been received. Our research team will reach out soon.")
            return redirect('contact')
        else:
            messages.error(request, "Please fill in all required fields.")

    context = {
        'page_title': 'Contact & Support | Deepfake Detector'
    }
    return render(request, 'detection/contact.html', context)


def about_view(request):
    """
    Informational page explaining the project, technology, and research background.
    """
    context = {
        'page_title': 'About Project | Deepfake Detection System'
    }
    return render(request, 'detection/about.html', context)


def how_it_works_view(request):
    """
    Explains the 5-step detection pipeline: Upload, Preprocess, Detect Face, AI Model, Result.
    """
    context = {
        'page_title': 'How It Works | Deepfake Detection Pipeline'
    }
    return render(request, 'detection/how_it_works.html', context)


@login_required
def dashboard_view(request):
    """
    User dashboard showing personal detection analytics, recent scans, and quick actions.
    """
    user_results = DetectionResult.objects.filter(user=request.user)
    
    total_detections = user_results.count()
    real_count = user_results.filter(result='REAL').count()
    fake_count = user_results.filter(result='FAKE').count()
    no_face_count = user_results.filter(
        Q(processing_status='NO_FACE') | Q(processing_status='MODEL_UNAVAILABLE') | Q(result='UNCERTAIN')
    ).count()
    
    latest_result = user_results.first()
    recent_detections = user_results[:8]

    authenticity_rate = round((real_count / total_detections * 100), 1) if total_detections > 0 else 0

    context = {
        'page_title': 'User Dashboard | Deepfake Detection System',
        'total_detections': total_detections,
        'real_count': real_count,
        'fake_count': fake_count,
        'no_face_count': no_face_count,
        'authenticity_rate': authenticity_rate,
        'latest_result': latest_result,
        'recent_detections': recent_detections,
    }
    return render(request, 'detection/dashboard.html', context)


@login_required
def detect_upload_view(request):
    """
    Handles media file upload, validation, preprocessing, face extraction, and AI prediction.

    Detection Pipeline (two independent models):
      Step 1 — AI-Generated Image Detector (whole image, no face needed):
               Uses onnx-community/ai-image-detect-distilled-ONNX to detect fully
               synthetic AI-generated images (Stable Diffusion, DALL-E, Midjourney, etc.)
      Step 2 — Deepfake / Face-Swap Detector (face crops required):
               Uses prithivML-deepfake-detector-v1 (SigLIP) to detect face-swap
               and GAN manipulation artifacts in detected face regions.

    Four-state combined verdict logic:
      AI_GENERATED : AI-gen detector says AI-generated (regardless of deepfake result)
      FAKE         : Deepfake detector says FAKE and AI-gen detector says REAL/UNCERTAIN/UNAVAILABLE
      REAL         : Both detectors say REAL
      UNCERTAIN    : Models disagree or both are below confidence threshold
    """
    if request.method == 'POST':
        form = MediaUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['file']
            is_valid, err_msg, media_type = validate_uploaded_media(uploaded_file)

            if not is_valid:
                messages.error(request, err_msg)
                return render(request, 'detection/upload.html', {'form': form, 'page_title': 'Analyze Media'})

            file_size_mb = uploaded_file.size / (1024 * 1024)
            detection_record = DetectionResult.objects.create(
                user=request.user,
                file=uploaded_file,
                file_name=uploaded_file.name,
                file_type=media_type,
                file_size_mb=round(file_size_mb, 2),
                processing_status='PROCESSING'
            )

            media_abs_path = detection_record.file.path

            try:
                # ─────────────────────────────────────────────────────────────
                # STEP 1: AI-Generated Image Detection (whole image)
                # Only run on images — for video we skip AI-gen detection
                # and rely on the deepfake face-level detector on sampled frames.
                # ─────────────────────────────────────────────────────────────
                ai_gen_result = {'prediction': 'UNAVAILABLE', 'available': False,
                                 'ai_gen_probability': 0.0, 'real_probability': 0.0,
                                 'confidence': 0.0, 'engine_used': 'Not run (video)'}

                if media_type == 'IMAGE':
                    ai_gen_result = predict_ai_generated(media_abs_path)
                    if not ai_gen_result['available']:
                        print(f"[detect_upload_view] AI-gen detector unavailable: {ai_gen_result.get('error')}")

                # ─────────────────────────────────────────────────────────────
                # STEP 2: Face Detection + Deepfake/Manipulation Detection
                # ─────────────────────────────────────────────────────────────
                if media_type == 'IMAGE':
                    prep_result = process_image(media_abs_path)
                else:
                    prep_result = process_video(media_abs_path)

                if not prep_result['face_detected']:
                    # No face found — use AI-gen result as the sole signal
                    # If AI-gen says AI_GENERATED, keep that verdict even without a face
                    ai_gen_pred = ai_gen_result.get('prediction', 'UNAVAILABLE')

                    if ai_gen_pred == 'AI_GENERATED' and ai_gen_result.get('available'):
                        no_face_result = 'AI_GENERATED'
                        no_face_status = 'COMPLETED'
                        no_face_conf = ai_gen_result.get('confidence', 0.0)
                        no_face_reason = (
                            "No human face detected in this image. "
                            "The AI-generated image detector classified the whole image as "
                            f"likely AI-generated ({ai_gen_result.get('ai_gen_probability', 0.0):.1%} probability). "
                            "No deepfake face-swap analysis was possible without a detectable face."
                        )
                    else:
                        no_face_result = 'UNCERTAIN'
                        no_face_status = 'NO_FACE'
                        no_face_conf = 0.0
                        no_face_reason = "No detectable face found in uploaded media."

                    detection_record.processing_status = no_face_status
                    detection_record.result = no_face_result
                    detection_record.confidence = no_face_conf
                    detection_record.face_detected = False
                    detection_record.faces_count = 0
                    detection_record.uncertainty_reason = no_face_reason
                    detection_record.error_message = prep_result.get(
                        'error_message', no_face_reason
                    )
                    detection_record.ai_generated_probability = ai_gen_result.get('ai_gen_probability', 0.0)
                    detection_record.ai_gen_model_name = ai_gen_result.get('engine_used', '')
                    detection_record.ai_gen_model_available = ai_gen_result.get('available', False)
                    detection_record.save()

                    messages.warning(
                        request,
                        "No recognizable face was detected. Please upload media with a visible face."
                    )
                    return redirect('detection_result', pk=detection_record.pk)

                # Face detected — run deepfake detector on face crops
                detection_record.face_detected = True
                detection_record.faces_count = prep_result['face_count']
                if prep_result.get('annotated_image_path'):
                    detection_record.extracted_face = prep_result['annotated_image_path']
                if 'frames_analyzed' in prep_result:
                    detection_record.frames_analyzed = prep_result['frames_analyzed']

                normalized_faces = prep_result.get('normalized_faces', [])
                face_meta_list = prep_result.get('face_metadata_list', [])

                deepfake_data = predict_deepfake(
                    normalized_faces,
                    media_path=media_abs_path,
                    media_type=media_type,
                    face_metadata_list=face_meta_list
                )

                # ─────────────────────────────────────────────────────────────
                # STEP 3: Four-State Combined Verdict Logic
                # ─────────────────────────────────────────────────────────────
                deepfake_verdict = deepfake_data['prediction']  # REAL | FAKE | UNCERTAIN
                ai_gen_verdict = ai_gen_result.get('prediction', 'UNAVAILABLE')  # AI_GENERATED | REAL | UNCERTAIN | UNAVAILABLE
                ai_gen_available = ai_gen_result.get('available', False)
                ai_gen_prob = ai_gen_result.get('ai_gen_probability', 0.0)

                if ai_gen_available and ai_gen_verdict == 'AI_GENERATED':
                    # AI-gen detector classified whole image as synthetic
                    combined_verdict = 'AI_GENERATED'
                    combined_confidence = ai_gen_result.get('confidence', 0.0)
                    combined_fake_prob = round(ai_gen_prob, 4)
                    combined_real_prob = round(1.0 - ai_gen_prob, 4)
                    combined_status = 'COMPLETED'
                    combined_explanation = (
                        f"AI-generated image detector classified this image as likely fully "
                        f"synthetic ({ai_gen_prob:.1%} probability). "
                        f"Deepfake face-level detector score: {deepfake_data.get('fake_probability', 0.0):.1%} manipulation probability. "
                        "Note: These are probabilistic estimates based on learned visual artifacts."
                    )

                elif deepfake_verdict == 'FAKE':
                    # Deepfake detector detected facial manipulation or global synthesis
                    combined_verdict = 'FAKE'
                    combined_confidence = deepfake_data['confidence']
                    combined_fake_prob = deepfake_data['fake_probability']
                    combined_real_prob = deepfake_data['real_probability']
                    combined_status = deepfake_data.get('analysis_status', 'COMPLETED')
                    combined_explanation = deepfake_data.get('explanation', '')
                    if ai_gen_available:
                        combined_explanation += (
                            f" AI-gen whole-image scan: {ai_gen_verdict} "
                            f"({ai_gen_prob:.1%} AI-gen probability)."
                        )

                elif deepfake_verdict == 'REAL' and (not ai_gen_available or ai_gen_verdict == 'REAL'):
                    # Both models agree the image is authentic
                    combined_verdict = 'REAL'
                    combined_confidence = deepfake_data['confidence']
                    combined_fake_prob = deepfake_data['fake_probability']
                    combined_real_prob = deepfake_data['real_probability']
                    combined_status = deepfake_data.get('analysis_status', 'COMPLETED')
                    combined_explanation = deepfake_data.get('explanation', '')
                    if ai_gen_available:
                        combined_explanation += (
                            f" AI-gen whole-image scan: {ai_gen_verdict} "
                            f"({ai_gen_prob:.1%} AI-gen probability)."
                        )

                else:
                    # Signals disagree, or one is uncertain (e.g., face looks smooth but whole image is ambiguous)
                    combined_verdict = 'UNCERTAIN'
                    combined_confidence = round(
                        abs(max(deepfake_data.get('fake_probability', 0.5), ai_gen_prob) - 0.5) * 200.0, 2
                    )
                    combined_fake_prob = deepfake_data.get('fake_probability', 0.0)
                    combined_real_prob = deepfake_data.get('real_probability', 0.0)
                    combined_status = deepfake_data.get('analysis_status', 'COMPLETED')
                    combined_explanation = (
                        f"Deepfake detector: {deepfake_verdict} ({deepfake_data.get('fake_probability', 0.0):.1%}) | "
                        f"AI-gen detector: {ai_gen_verdict} ({ai_gen_prob:.1%}). "
                        "Signals are in the borderline uncertainty region — cannot certify authenticity."
                    )

                # ─────────────────────────────────────────────────────────────
                # STEP 4: Persist combined results
                # ─────────────────────────────────────────────────────────────
                detection_record.result = combined_verdict
                detection_record.processing_status = combined_status
                detection_record.confidence = combined_confidence
                detection_record.fake_probability = combined_fake_prob
                detection_record.real_probability = combined_real_prob
                detection_record.model_name = deepfake_data.get('engine_used', 'CNN Deepfake Classifier')
                detection_record.uncertainty_reason = combined_explanation
                detection_record.error_message = combined_explanation

                # AI-gen specific fields
                detection_record.ai_generated_probability = ai_gen_result.get('ai_gen_probability', 0.0)
                detection_record.ai_gen_model_name = ai_gen_result.get('engine_used', '')
                detection_record.ai_gen_model_available = ai_gen_available

                forensic_details = deepfake_data.get('forensic_details', {})
                forensic_details['is_heuristic'] = deepfake_data.get('is_heuristic', False)
                forensic_details['model_available'] = deepfake_data.get('model_available', True)
                forensic_details['faces_analysis'] = deepfake_data.get('faces_analysis', [])
                forensic_details['ai_gen_result'] = {
                    'prediction': ai_gen_verdict,
                    'ai_gen_probability': ai_gen_result.get('ai_gen_probability', 0.0),
                    'real_probability': ai_gen_result.get('real_probability', 0.0),
                    'confidence': ai_gen_result.get('confidence', 0.0),
                    'engine': ai_gen_result.get('engine_used', ''),
                    'available': ai_gen_available,
                    'error': ai_gen_result.get('error'),
                }
                detection_record.forensic_details = forensic_details
                detection_record.save()

                # Save relational FaceDetectionItem records
                for f_item in deepfake_data.get('faces_analysis', []):
                    bbox = f_item.get('bbox', [0, 0, 0, 0])
                    signals = f_item.get('forensic_signals', {})
                    FaceDetectionItem.objects.create(
                        detection_result=detection_record,
                        face_number=f_item.get('face_id', 1),
                        bounding_box_x=bbox[0] if len(bbox) > 0 else 0,
                        bounding_box_y=bbox[1] if len(bbox) > 1 else 0,
                        bounding_box_w=bbox[2] if len(bbox) > 2 else 0,
                        bounding_box_h=bbox[3] if len(bbox) > 3 else 0,
                        face_crop=f_item.get('crop_path'),
                        label=f_item.get('model_prediction', 'UNCERTAIN'),
                        confidence=f_item.get('confidence', 0.0),
                        real_score=f_item.get('real_probability', 0.0),
                        fake_score=f_item.get('fake_probability', 0.0),
                        frame_number=f_item.get('frame_number', 1),
                        frame_timestamp=f_item.get('frame_timestamp', 0.0),
                        laplacian_variance=signals.get('laplacian_variance', 0.0),
                        frequency_energy=signals.get('frequency_energy', 0.0),
                        noise_estimate=signals.get('noise_estimate', 0.0),
                    )

                # User-facing flash messages
                if combined_status == 'MODEL_UNAVAILABLE':
                    messages.warning(
                        request,
                        "Analysis Inconclusive: A validated deepfake classifier is not available. Displaying supporting forensic measurements only."
                    )
                elif combined_verdict == 'AI_GENERATED':
                    messages.warning(
                        request,
                        f"AI-Generated Image Detected with {combined_confidence:.1f}% confidence. "
                        "This image appears to be fully synthetic (not a real photograph)."
                    )
                elif combined_verdict == 'UNCERTAIN':
                    messages.info(
                        request,
                        "Analysis Inconclusive: Evidence falls within the uncertainty region."
                    )
                else:
                    messages.success(
                        request,
                        f"Analysis complete. Content classified as {combined_verdict} with {detection_record.confidence:.1f}% confidence."
                    )
                return redirect('detection_result', pk=detection_record.pk)

            except Exception as e:
                import traceback
                print(f"[detect_upload_view] Unexpected error: {e}\n{traceback.format_exc()}")
                detection_record.processing_status = 'FAILED'
                detection_record.error_message = f"An unexpected processing error occurred: {str(e)}"
                detection_record.save()
                messages.error(request, "Failed to analyze the media file. Please try again with a standard format.")
                return redirect('detection_result', pk=detection_record.pk)

        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = MediaUploadForm()

    context = {
        'form': form,
        'page_title': 'Analyze Media | Upload Image or Video'
    }
    return render(request, 'detection/upload.html', context)


@login_required
def result_view(request, pk):
    """
    Displays the detailed detection results, confidence visualization, and forensic report.
    """
    if request.user.is_staff:
        result = get_object_or_404(DetectionResult, pk=pk)
    else:
        result = get_object_or_404(DetectionResult, pk=pk, user=request.user)

    face_items = result.face_items.all()

    context = {
        'result': result,
        'face_items': face_items,
        'page_title': f"Detection Result: {result.result} ({result.file_name})"
    }
    return render(request, 'detection/result.html', context)


@login_required
def history_view(request):
    """
    Displays historical detection records for the logged-in user with filter and pagination.
    """
    user_results = DetectionResult.objects.filter(user=request.user)

    filter_type = request.GET.get('filter', 'all')
    if filter_type == 'real':
        user_results = user_results.filter(result='REAL')
    elif filter_type == 'fake':
        user_results = user_results.filter(result='FAKE')
    elif filter_type == 'ai_generated':
        user_results = user_results.filter(result='AI_GENERATED')
    elif filter_type == 'no_face':
        user_results = user_results.filter(
            Q(processing_status='NO_FACE') | Q(processing_status='MODEL_UNAVAILABLE') | Q(result='UNCERTAIN')
        )

    search_query = request.GET.get('q', '').strip()
    if search_query:
        user_results = user_results.filter(file_name__icontains=search_query)

    paginator = Paginator(user_results, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'filter_type': filter_type,
        'search_query': search_query,
        'total_count': user_results.count(),
        'page_title': 'Detection History | Previous Scans'
    }
    return render(request, 'detection/history.html', context)


@login_required
def details_view(request, pk):
    """
    Detailed inspection page for a historical scan record.
    """
    if request.user.is_staff:
        result = get_object_or_404(DetectionResult, pk=pk)
    else:
        result = get_object_or_404(DetectionResult, pk=pk, user=request.user)

    face_items = result.face_items.all()

    context = {
        'result': result,
        'face_items': face_items,
        'page_title': f"Scan Details: {result.file_name}"
    }
    return render(request, 'detection/details.html', context)


@login_required
def delete_result_view(request, pk):
    """
    Deletes a specific detection record and its associated files.
    """
    if request.user.is_staff:
        result = get_object_or_404(DetectionResult, pk=pk)
    else:
        result = get_object_or_404(DetectionResult, pk=pk, user=request.user)

    if request.method == 'POST':
        file_name = result.file_name
        result.delete()
        messages.success(request, f"Detection record for '{file_name}' has been deleted.")
        return redirect('history')

    return redirect('history')


@login_required
def report_view(request, pk):
    """
    Generates and serves a downloadable PDF forensic report, logging the report download.
    """
    if request.user.is_staff:
        result = get_object_or_404(DetectionResult, pk=pk)
    else:
        result = get_object_or_404(DetectionResult, pk=pk, user=request.user)

    report_id = f"DFR-{result.pk:05d}-{timezone.now().strftime('%Y%m%d')}"
    report_record, created = AnalysisReport.objects.get_or_create(
        detection_result=result,
        report_id=report_id,
        defaults={'generated_by': request.user}
    )
    if not created:
        report_record.download_count += 1
        report_record.save(update_fields=['download_count'])

    return generate_pdf_report(result, report_id=report_id)


@staff_member_required
def admin_dashboard_view(request):
    """
    Dedicated Administrator Dashboard showing platform-wide statistics,
    user activity, verification distributions, and moderation actions.
    """
    total_users = User.objects.count()
    total_detections = DetectionResult.objects.count()
    total_real = DetectionResult.objects.filter(result='REAL').count()
    total_fake = DetectionResult.objects.filter(result='FAKE').count()
    total_no_face = DetectionResult.objects.filter(
        Q(processing_status='NO_FACE') | Q(processing_status='MODEL_UNAVAILABLE') | Q(result='UNCERTAIN')
    ).count()
    total_images = DetectionResult.objects.filter(file_type='IMAGE').count()
    total_videos = DetectionResult.objects.filter(file_type='VIDEO').count()

    total_storage = DetectionResult.objects.aggregate(total_size=Sum('file_size_mb'))['total_size'] or 0.0

    recent_activity = DetectionResult.objects.select_related('user').all()[:20]
    all_users = User.objects.all().order_by('-date_joined')[:15]

    context = {
        'page_title': 'Admin Statistics Dashboard | Deepfake Detection System',
        'total_users': total_users,
        'total_detections': total_detections,
        'total_real': total_real,
        'total_fake': total_fake,
        'total_no_face': total_no_face,
        'total_images': total_images,
        'total_videos': total_videos,
        'total_storage': round(total_storage, 2),
        'recent_activity': recent_activity,
        'all_users': all_users,
    }
    return render(request, 'detection/admin_dashboard.html', context)
