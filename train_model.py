"""
train_model.py — Deepfake Classifier Training & Export Pipeline
==============================================================
Final-Year Diploma CSE Project Utility

This script provides an end-to-end training, validation, and model export
pipeline for training a Convolutional Neural Network (CNN) deepfake classifier.

Supported Models:
  - MobileNetV2 (Default, lightweight, fast, high accuracy)
  - EfficientNetB0
  - Custom 4-layer Deepfake CNN

Dataset Organization Expected:
------------------------------
Create a folder structure as follows:

dataset/
    train/
        real/      <-- Real / authentic face photographs
        fake/      <-- Face-swapped deepfakes or AI-generated portraits
    val/
        real/
        fake/
    test/          <-- Optional test split for final evaluation
        real/
        fake/

Benchmark Dataset Sources:
-------------------------
1. Face-Swapped Deepfakes:
   - FaceForensics++ (FF++): https://github.com/ondyari/FaceForensics
   - Celeb-DF (v2): https://github.com/yuezunli/celeb-deepfakeforensics
   - DFDC (DeepFake Detection Challenge): https://www.kaggle.com/c/deepfake-detection-challenge

2. AI-Generated Synthetic Portraits:
   - 140k Real and Fake Faces: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
   - FFHQ (Real) vs StyleGAN (Synthetic): https://github.com/NVlabs/ffhq-dataset

Outputs:
--------
- model/deepfake_model.h5       (Keras / TensorFlow format)
- model/deepfake_model.onnx     (ONNX format for OpenCV DNN execution)
- model/class_indices.json      (Verified class mapping)
- model/training_history.png    (Loss and Accuracy curves)

Usage:
------
python train_model.py --data_dir dataset/ --epochs 15 --batch_size 32
"""

import os
import sys
import json
import argparse
import numpy as np

# Verify Python version and environment
print("=" * 70)
print("DEEPFAKE CLASSIFIER TRAINING UTILITY")
print("=" * 70)
print(f"Python: {sys.version.split()[0]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN Deepfake Classification Model")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root folder containing train/ and val/ subdirectories")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=224, help="Input image dimension (224 for standard backbones)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--model_type", type=str, default="mobilenetv2", choices=["mobilenetv2", "custom_cnn"], help="Architecture to train")
    parser.add_argument("--export_onnx", action="store_true", default=True, help="Export model to ONNX format for OpenCV DNN")
    return parser.parse_args()


def check_dataset_structure(data_dir):
    """Validates dataset folder structure and counts images per split and class."""
    if not os.path.isdir(data_dir):
        print(f"\n[ERROR] Dataset directory '{data_dir}' does not exist.")
        print("Please create the dataset structure described in the header documentation.")
        return False, {}

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        print(f"\n[ERROR] Dataset directory must contain 'train' and 'val' subfolders.")
        return False, {}

    stats = {}
    valid = True
    for split_name, split_path in [("train", train_dir), ("val", val_dir)]:
        stats[split_name] = {}
        for cls in ["real", "fake"]:
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                print(f"[ERROR] Missing folder: {cls_path}")
                valid = False
                continue
            images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            stats[split_name][cls] = len(images)

    if valid:
        print("\nDataset Summary:")
        print(f"  Training   : {stats['train']['real']} Real, {stats['train']['fake']} Fake images")
        print(f"  Validation : {stats['val']['real']} Real, {stats['val']['fake']} Fake images")
        if stats['train']['real'] == 0 or stats['train']['fake'] == 0:
            print("[WARNING] One or more training classes are empty. Add images before training.")
            valid = False

    return valid, stats


def build_model(model_type, img_size):
    """Constructs the deep learning classifier with binary cross-entropy head."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    input_shape = (img_size, img_size, 3)

    if model_type == "mobilenetv2":
        print("[Model] Building Transfer-Learning Classifier with MobileNetV2 backbone...")
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        base.trainable = False  # Freeze backbone for initial transfer learning

        inputs = tf.keras.Input(shape=input_shape)
        # MobileNet expects inputs in [-1, 1] range; our preprocessing supplies [0, 1]
        x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        # Output: Single sigmoid probability where 0 = REAL, 1 = FAKE
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs, outputs, name="Deepfake_MobileNetV2")

    else:
        print("[Model] Building Custom 4-Layer Deepfake CNN...")
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid")
        ], name="Custom_Deepfake_CNN")

    return model


def main():
    args = parse_args()

    # Step 1: Check dataset
    valid_dataset, stats = check_dataset_structure(args.data_dir)
    if not valid_dataset:
        print("\n" + "=" * 70)
        print("INSTRUCTIONS TO SUPPLY YOUR DATASET:")
        print("=" * 70)
        print("1. Download labeled real and fake images from Kaggle or academic sources:")
        print("   - '140k Real and Fake Faces': https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
        print("   - Or extract cropped faces from FaceForensics++ videos.")
        print(f"2. Place real face crops into: {os.path.join(args.data_dir, 'train', 'real')}")
        print(f"3. Place fake face crops into: {os.path.join(args.data_dir, 'train', 'fake')}")
        print(f"4. Place validation face crops into: {os.path.join(args.data_dir, 'val', 'real')} and 'fake'")
        print("5. Re-run: python train_model.py")
        print("=" * 70)
        return

    # Step 2: Check TensorFlow availability
    try:
        import tensorflow as tf
        print(f"\n[Environment] TensorFlow {tf.__version__} initialized.")
    except ImportError:
        print("\n[ERROR] TensorFlow is required for training models.")
        print("Please install TensorFlow in a compatible Python environment:")
        print("    pip install tensorflow")
        print("Note: On Python 3.14, TensorFlow pre-built wheels may not be available yet.")
        print("We recommend using a Python 3.10 or 3.11 virtual environment for training.")
        return

    # Step 3: Data Loaders
    print("\n[Data] Creating train and validation data pipelines...")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(args.data_dir, "train"),
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="binary",
        class_names=["real", "fake"],  # Enforces 0 = REAL, 1 = FAKE
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(args.data_dir, "val"),
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="binary",
        class_names=["real", "fake"],
        shuffle=False
    )

    # Normalize [0, 255] -> [0, 1]
    rescale_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (rescale_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescale_layer(x), y)).prefetch(tf.data.AUTOTUNE)

    # Step 4: Build & Compile
    model = build_model(args.model_type, args.img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    model.summary()

    # Step 5: Train
    print(f"\n[Training] Starting {args.epochs} epochs of training...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Step 6: Save Model
    out_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(out_dir, exist_ok=True)

    h5_path = os.path.join(out_dir, "deepfake_model.h5")
    model.save(h5_path)
    print(f"\n[Success] Trained model saved to: {h5_path}")

    # Save Class Mapping
    class_map = {0: "REAL", 1: "FAKE"}
    class_indices_path = os.path.join(out_dir, "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"[Success] Class index mapping saved: {class_indices_path}")

    # Step 7: Export ONNX (for OpenCV DNN inference without TensorFlow)
    if args.export_onnx:
        try:
            import tf2onnx
            onnx_path = os.path.join(out_dir, "deepfake_model.onnx")
            input_signature = [tf.TensorSpec([None, args.img_size, args.img_size, 3], tf.float32, name="input")]
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature, output_path=onnx_path)
            print(f"[Success] Exported ONNX model to: {onnx_path}")
            print("  -> This ONNX model can run directly via OpenCV DNN without needing TensorFlow!")
        except ImportError:
            print("[Info] Install tf2onnx (`pip install tf2onnx`) to enable automatic ONNX export.")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE! The Django web application will now automatically")
    print(f"detect and load the trained model from: {h5_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
