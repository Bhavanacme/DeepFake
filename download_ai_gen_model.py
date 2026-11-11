"""
Download AI-Generated Image Detection Model
============================================
Downloads the onnx-community/ai-image-detect-distilled-ONNX model
into model/ai_gen_detector/ for use by the Deepfake Detection System.

License: Apache-2.0 (onnx-community/ai-image-detect-distilled-ONNX)

Usage:
    python download_ai_gen_model.py

Requirements: pip install huggingface_hub
"""

import os
import sys

# Allow running outside Django context
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_GEN_MODEL_DIR = os.path.join(BASE_DIR, 'model', 'ai_gen_detector')

REPO_ID = 'onnx-community/ai-image-detect-distilled-ONNX'
FILES = [
    ('onnx/model.onnx', 'model.onnx'),
    ('config.json', 'config.json'),
    ('preprocessor_config.json', 'preprocessor_config.json'),
]


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed.")
        print("        Run: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(AI_GEN_MODEL_DIR, exist_ok=True)
    print(f"Target directory: {AI_GEN_MODEL_DIR}\n")

    success = True
    for repo_file, local_name in FILES:
        local_dest = os.path.join(AI_GEN_MODEL_DIR, local_name)
        if os.path.isfile(local_dest):
            size_mb = os.path.getsize(local_dest) / (1024 * 1024)
            print(f"[SKIP] {local_name} already exists ({size_mb:.1f} MB)")
            continue

        print(f"[DOWNLOAD] {repo_file} -> {local_dest}")
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_file,
                local_dir=AI_GEN_MODEL_DIR,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download may nest the file in a subfolder; copy to flat layout
            if os.path.isfile(downloaded) and os.path.abspath(downloaded) != os.path.abspath(local_dest):
                import shutil
                shutil.copy2(downloaded, local_dest)
                print(f"         Copied to: {local_dest}")
            size_mb = os.path.getsize(local_dest) / (1024 * 1024) if os.path.isfile(local_dest) else 0
            print(f"         Done ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[ERROR] Failed to download {repo_file}: {e}")
            success = False

    print()
    if success:
        model_path = os.path.join(AI_GEN_MODEL_DIR, 'model.onnx')
        if os.path.isfile(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"[OK] AI-gen model ready: {model_path} ({size_mb:.1f} MB)")
            print("[OK] Run 'python manage.py test detection' to verify integration.")
        else:
            print("[WARNING] model.onnx was not found after download. Check for errors above.")
    else:
        print("[FAILED] Some downloads failed. Check your network connection and retry.")
        sys.exit(1)


if __name__ == '__main__':
    main()
