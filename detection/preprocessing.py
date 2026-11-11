import os
import cv2
import numpy as np
from PIL import Image
from django.conf import settings
from django.utils import timezone


ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

# Path to YuNet ONNX model (OpenCV 5 face detector)
YUNET_MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'face_detection_yunet_2023mar.onnx')


def validate_uploaded_media(uploaded_file):
    """
    Validates uploaded file for extension, size, and integrity.
    Returns (is_valid: bool, error_message: str or None, media_type: str or None).
    """
    if not uploaded_file:
        return False, "No file was uploaded. Please select an image or video.", None

    name = uploaded_file.name.lower()
    _, ext = os.path.splitext(name)

    if uploaded_file.size == 0:
        return False, "The selected file is empty (0 bytes). Please upload a valid media file.", None

    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        size_mb = uploaded_file.size / (1024 * 1024)
        return False, f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of 50 MB.", None

    if ext in ALLOWED_IMAGE_EXTENSIONS:
        try:
            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
            img.verify()
            uploaded_file.seek(0)
        except Exception:
            return False, "The uploaded image file appears corrupted or unreadable.", None
        return True, None, "IMAGE"

    elif ext in ALLOWED_VIDEO_EXTENSIONS:
        return True, None, "VIDEO"

    else:
        allowed_all = ", ".join(sorted(list(ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS)))
        return False, f"Unsupported file format '{ext}'. Allowed formats: {allowed_all}", None


def get_yunet_detector(img_width=320, img_height=320):
    """
    Creates and returns an OpenCV YuNet (FaceDetectorYN) face detector.
    """
    if not os.path.exists(YUNET_MODEL_PATH):
        raise FileNotFoundError(
            f"YuNet face detection model not found at: {YUNET_MODEL_PATH}\n"
            f"Please ensure 'face_detection_yunet_2023mar.onnx' is placed in the 'model/' directory."
        )

    detector = cv2.FaceDetectorYN_create(
        model=YUNET_MODEL_PATH,
        config='',
        input_size=(img_width, img_height),
        score_threshold=0.55,
        nms_threshold=0.3,
        top_k=5000
    )
    return detector


def read_image_from_path(image_path):
    """
    Safely reads an image from a file path, handling Windows Unicode path issues.
    Returns an OpenCV BGR image (np.ndarray) or None on failure.
    """
    try:
        with open(image_path, 'rb') as f:
            bytes_data = f.read()
        np_arr = np.frombuffer(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ImageRead] Error reading image: {e}")
        return cv2.imread(image_path)


def detect_faces_yunet(img):
    """
    Detects faces in an OpenCV BGR image using YuNet (FaceDetectorYN).
    Returns a list of (x, y, w, h) bounding boxes, sorted by area descending.
    """
    h_orig, w_orig = img.shape[:2]

    max_dim = 1280
    scale = 1.0
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img
        new_w, new_h = w_orig, h_orig

    detector = get_yunet_detector(img_width=new_w, img_height=new_h)
    detector.setInputSize((new_w, new_h))

    retval, faces_raw = detector.detect(img_resized)

    if faces_raw is None or len(faces_raw) == 0:
        # Fallback with lower threshold
        detector2 = cv2.FaceDetectorYN_create(
            model=YUNET_MODEL_PATH,
            config='',
            input_size=(new_w, new_h),
            score_threshold=0.35,
            nms_threshold=0.3,
            top_k=5000
        )
        detector2.setInputSize((new_w, new_h))
        retval, faces_raw = detector2.detect(img_resized)
        if faces_raw is None or len(faces_raw) == 0:
            return []

    boxes = []
    for face in faces_raw:
        x = int(face[0] / scale)
        y = int(face[1] / scale)
        w = int(face[2] / scale)
        h = int(face[3] / scale)

        x = max(0, x)
        y = max(0, y)
        w = min(w, w_orig - x)
        h = min(h, h_orig - y)

        if w > 0 and h > 0:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def process_image(image_path, relative_save_dir="processed/faces/"):
    """
    Preprocesses an image file:
    1. Reads and standardizes image
    2. Detects faces using YuNet
    3. Crops faces, saves crops & normalizes for model input
    4. Annotates original image with modern bounding boxes
    """
    img = read_image_from_path(image_path)
    if img is None:
        return {
            'success': False,
            'face_detected': False,
            'face_count': 0,
            'primary_face_crop': None,
            'annotated_image_path': None,
            'normalized_faces': [],
            'face_metadata_list': [],
            'error_message': "Could not decode or open the image file."
        }

    h, w = img.shape[:2]
    faces = detect_faces_yunet(img)

    if len(faces) == 0:
        return {
            'success': True,
            'face_detected': False,
            'face_count': 0,
            'primary_face_crop': None,
            'annotated_image_path': None,
            'normalized_faces': [],
            'face_metadata_list': [],
            'error_message': "No recognizable human face was detected in this image. Please upload a clear photo with a visible, well-lit face."
        }

    annotated = img.copy()
    normalized_faces = []
    face_metadata_list = []
    primary_crop = None

    now_stamp = timezone.now().strftime("%Y%m%d_%H%M%S_%f")
    full_save_dir = os.path.join(settings.MEDIA_ROOT, relative_save_dir)
    os.makedirs(full_save_dir, exist_ok=True)

    for idx, (x, y, fw, fh) in enumerate(faces):
        margin_x = int(fw * 0.15)
        margin_y = int(fh * 0.15)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        if primary_crop is None:
            primary_crop = face_crop

        face_crop_filename = f"crop_{now_stamp}_face_{idx+1}.jpg"
        face_crop_full_path = os.path.join(full_save_dir, face_crop_filename)
        cv2.imwrite(face_crop_full_path, face_crop)
        face_crop_rel_path = os.path.join(relative_save_dir, face_crop_filename).replace('\\', '/')

        face_resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
        face_normalized = face_resized.astype(np.float32) / 255.0
        normalized_faces.append(face_normalized)

        face_metadata_list.append({
            'face_number': idx + 1,
            'bbox': [x, y, fw, fh],
            'normalized_crop': face_normalized,
            'crop_path': face_crop_rel_path,
            'frame_number': 1,
            'frame_timestamp': 0.0,
        })

        # Draw futuristic bounding box
        box_color = (255, 180, 0) if idx == 0 else (200, 200, 200)
        cv2.rectangle(annotated, (x, y), (x + fw, y + fh), box_color, 2)
        line_len = max(10, int(fw * 0.15))
        thickness = 3
        cv2.line(annotated, (x, y), (x + line_len, y), box_color, thickness)
        cv2.line(annotated, (x, y), (x, y + line_len), box_color, thickness)
        cv2.line(annotated, (x + fw, y), (x + fw - line_len, y), box_color, thickness)
        cv2.line(annotated, (x + fw, y), (x + fw, y + line_len), box_color, thickness)
        cv2.line(annotated, (x, y + fh), (x + line_len, y + fh), box_color, thickness)
        cv2.line(annotated, (x, y + fh), (x, y + fh - line_len), box_color, thickness)
        cv2.line(annotated, (x + fw, y + fh), (x + fw - line_len, y + fh), box_color, thickness)
        cv2.line(annotated, (x + fw, y + fh), (x + fw, y + fh - line_len), box_color, thickness)

        label = f"Face #{idx + 1}"
        cv2.putText(
            annotated, label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            box_color, 2, cv2.LINE_AA
        )

    if not normalized_faces:
        return {
            'success': True,
            'face_detected': False,
            'face_count': 0,
            'primary_face_crop': None,
            'annotated_image_path': None,
            'normalized_faces': [],
            'face_metadata_list': [],
            'error_message': "Face bounding boxes were detected but crops were invalid."
        }

    save_filename = f"detected_face_{now_stamp}.jpg"
    full_save_path = os.path.join(full_save_dir, save_filename)
    cv2.imwrite(full_save_path, annotated)
    relative_path = os.path.join(relative_save_dir, save_filename).replace('\\', '/')

    return {
        'success': True,
        'face_detected': True,
        'face_count': len(faces),
        'primary_face_crop': primary_crop,
        'annotated_image_path': relative_path,
        'normalized_faces': normalized_faces,
        'face_metadata_list': face_metadata_list,
        'error_message': None
    }


def process_video(video_path, relative_save_dir="processed/faces/", max_frames_to_sample=12):
    """
    Preprocesses a video file:
    1. Extracts evenly spaced frames across the video timeline with timestamps
    2. Performs face detection on extracted frames
    3. Gathers face crops for batch model inference
    4. Saves an annotated representative preview frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'success': False,
            'face_detected': False,
            'face_count': 0,
            'frames_analyzed': 0,
            'annotated_image_path': None,
            'normalized_faces': [],
            'face_metadata_list': [],
            'error_message': "Unable to open or read video file. Video might be corrupted or codec unsupported."
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total_frames <= 0:
        total_frames = 100

    sample_count = min(max_frames_to_sample, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int)

    detected_faces_list = []
    face_metadata_list = []
    best_annotated_frame = None
    best_face_area = 0
    frames_processed_count = 0

    now_stamp = timezone.now().strftime("%Y%m%d_%H%M%S_%f")
    full_save_dir = os.path.join(settings.MEDIA_ROOT, relative_save_dir)
    os.makedirs(full_save_dir, exist_ok=True)

    face_global_idx = 0
    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frames_processed_count += 1
        h, w = frame.shape[:2]
        timestamp_sec = round(float(f_idx) / fps, 2)

        faces = detect_faces_yunet(frame)

        for (x, y, fw, fh) in faces:
            area = fw * fh
            margin_x = int(fw * 0.15)
            margin_y = int(fh * 0.15)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + fw + margin_x)
            y2 = min(h, y + fh + margin_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                face_global_idx += 1
                resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                normalized = resized.astype(np.float32) / 255.0
                detected_faces_list.append(normalized)

                face_crop_filename = f"video_face_{now_stamp}_f{f_idx}_{face_global_idx}.jpg"
                face_crop_full_path = os.path.join(full_save_dir, face_crop_filename)
                cv2.imwrite(face_crop_full_path, crop)
                face_crop_rel_path = os.path.join(relative_save_dir, face_crop_filename).replace('\\', '/')

                face_metadata_list.append({
                    'face_number': face_global_idx,
                    'bbox': [x, y, fw, fh],
                    'normalized_crop': normalized,
                    'crop_path': face_crop_rel_path,
                    'frame_number': int(f_idx),
                    'frame_timestamp': timestamp_sec,
                })

            if area > best_face_area:
                best_face_area = area
                ann_frame = frame.copy()
                box_color = (255, 180, 0)
                cv2.rectangle(ann_frame, (x, y), (x + fw, y + fh), box_color, 2)
                cv2.putText(
                    ann_frame, f"Face Detected (Frame {f_idx} @ {timestamp_sec}s)",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    box_color, 2, cv2.LINE_AA
                )
                best_annotated_frame = ann_frame

    cap.release()

    if not detected_faces_list or best_annotated_frame is None:
        return {
            'success': True,
            'face_detected': False,
            'face_count': 0,
            'frames_analyzed': frames_processed_count,
            'annotated_image_path': None,
            'normalized_faces': [],
            'face_metadata_list': [],
            'error_message': "No recognizable human faces detected across video frames. Please ensure the video features a visible face."
        }

    save_filename = f"video_keyframe_{now_stamp}.jpg"
    full_save_path = os.path.join(full_save_dir, save_filename)
    cv2.imwrite(full_save_path, best_annotated_frame)
    relative_path = os.path.join(relative_save_dir, save_filename).replace('\\', '/')

    return {
        'success': True,
        'face_detected': True,
        'face_count': len(detected_faces_list),
        'frames_analyzed': frames_processed_count,
        'annotated_image_path': relative_path,
        'normalized_faces': detected_faces_list,
        'face_metadata_list': face_metadata_list,
        'error_message': None
    }
