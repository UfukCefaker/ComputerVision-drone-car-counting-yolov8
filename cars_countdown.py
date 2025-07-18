# %% [markdown]
# # Computer Vision Lab – Assignment 3
# 
# **Name:** Ufuk Cefaker  
# **Student ID:** b2220356171

# %% [markdown]
# # **Part 1: Dataset Preparation for YOLOv8**
# 
# 

# %%
# YOLOv8 için ultralytics paketi
!pip install -q ultralytics

# Temel kütüphaneler
import os
import shutil
import zipfile
import glob
from pathlib import Path

# %%
!gdown --id 1RwB_X8SxQnqoVT7IiTQD9GsX9-FRd8Ib --output cars_dataset.zip

with zipfile.ZipFile("cars_dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("cars_dataset")

# %%
import os
from pathlib import Path
import shutil

# Define paths
original_images = Path("cars_dataset/Images")
original_labels = Path("cars_dataset/Annotations")
splits_folder = Path("cars_dataset/ImageSets")

# Create YOLO format folders
yolo_base = Path("cars_yolo_dataset")
for folder in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
    (yolo_base / folder).mkdir(parents=True, exist_ok=True)

# Function to copy images and labels to target folders
def copy_split(split_name):
    # Read image names (without extension) from txt file
    with open(splits_folder / f"{split_name}.txt") as f:
        image_names = [line.strip() for line in f.readlines()]

    for name in image_names:
        img_path = original_images / f"{name}.png"  # assume .png
        label_path = original_labels / f"{name}.txt"

        # Copy if both image and label exist
        if img_path.exists() and label_path.exists():
            shutil.copy(img_path, yolo_base / f"images/{split_name}/{img_path.name}")
            shutil.copy(label_path, yolo_base / f"labels/{split_name}/{name}.txt")
        else:
            print(f"Skipping missing: {name}")

# Apply the function to all splits
for split in ["train", "val", "test"]:
    copy_split(split)

# %% [markdown]
# ### 📁 Dataset Conversion Description
# 
# This code block prepares the dataset structure required for training a YOLOv8 model:
# 
# - It reads image names from `train.txt`, `val.txt`, and `test.txt` located in the `ImageSets` folder.
# - For each image name, it checks for the existence of both the image file (`.png`) and its corresponding annotation file (`.txt`).
# - If both files exist, they are copied into the appropriate YOLOv8-compatible folder: `images/{split}` and `labels/{split}`.
# - If any file is missing, it skips the sample and prints a message indicating the missing file.
# - The folder structure created under `cars_yolo_dataset` follows the standard YOLOv8 directory format.
# 
# By the end of this process, the dataset is organized and ready for YOLOv8 training.

# %%
import os

abs_path = os.path.abspath("cars_yolo_dataset")

with open("data.yaml", "w") as f:
    f.write(f"""path: {abs_path}
train: images/train
val: images/val
test: images/test

names:
  0: car
""")

print("✅ data.yaml dosyası mutlak path ile güncellendi.")

# %% [markdown]
# This code generates a `data.yaml` file using the absolute path of the dataset directory. The file defines the dataset splits and class names required by YOLOv8 for training.

# %%
# Show the contents of the data.yaml file
!cat cars_yolo_dataset/data.yaml

# %%
from pathlib import Path
import os
from PIL import Image

# Base paths
image_base = Path("cars_dataset/Images")
label_base = Path("cars_dataset/Annotations")
split_base = Path("cars_dataset/ImageSets")
yolo_label_base = Path("cars_yolo_dataset/labels")
yolo_image_base = Path("cars_yolo_dataset/images")

# Create label folders if not exist
for split in ["train", "val", "test"]:
    (yolo_label_base / split).mkdir(parents=True, exist_ok=True)

# Process splits
for split in ["train", "val", "test"]:
    with open(split_base / f"{split}.txt") as f:
        image_names = [line.strip() for line in f.readlines()]

    for name in image_names:
        img_path = image_base / f"{name}.png"
        ann_path = label_base / f"{name}.txt"
        out_path = yolo_label_base / split / f"{name}.txt"

        if not img_path.exists() or not ann_path.exists():
            print(f"Skipping missing: {name}")
            continue

        # Get image dimensions
        with Image.open(img_path) as im:
            w, h = im.size

        new_lines = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                xmin, ymin, xmax, ymax = map(float, parts[:4])
                class_id = 0
                x_center = (xmin + xmax) / 2 / w
                y_center = (ymin + ymax) / 2 / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h
                if width <= 0 or height <= 0:
                    continue
                new_lines.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save converted label
        with open(out_path, "w") as f:
            f.write("\n".join(new_lines))

# %% [markdown]
# ### 🔹 Dataset Preparation for YOLOv8
# 
# In this step, we prepared the dataset from its raw form into a structure and format compatible with YOLOv8:
# 
# - The original dataset includes drone-captured images and annotation files.
# - We first read the image split lists (`train.txt`, `val.txt`, `test.txt`) from the `ImageSets` folder.
# - For each listed image:
#   - The corresponding `.png` image and `.txt` annotation are loaded (if available).
#   - The annotation in Pascal VOC format (`xmin, ymin, xmax, ymax`) is converted into YOLO format (`x_center, y_center, width, height`) by normalizing with the image width and height.
#   - The converted annotation is saved under `cars_yolo_dataset/labels/{split}/`.
# - Missing image or annotation files are skipped with a warning.
# 
# The resulting dataset follows the YOLOv8 folder structure and uses normalized YOLO labels. With this step completed, the dataset is ready for training.

# %% [markdown]
# # **Part2: Train models under four different freezing settings:**

# %%
from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO("yolov8n.pt")  # nano model, fast and light

# %%
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# 📈 Draw training and validation loss curves
def draw_loss_graph(experiment_name):
    results_path = os.path.join("runs/detect", experiment_name, "results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot training losses
        axes[0].plot(df["train/box_loss"], label="Box Loss")
        axes[0].plot(df["train/cls_loss"], label="Class Loss")
        axes[0].set_title("Training Loss by Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True)

        # Plot validation losses
        axes[1].plot(df["val/box_loss"], label="Box Loss")
        axes[1].plot(df["val/cls_loss"], label="Class Loss")
        axes[1].set_title("Validation Loss by Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print(f"❌ Could not find: {results_path}")

# 🔒 Print which layers are frozen before training
def print_frozen_layers(model, freeze_level):
    print("\n📌 Layer Freeze Status (Before Training):")
    modules = model.model.model  # Ultralytics internal model structure

    frozen_indices = list(range(freeze_level))
    frozen_param_count = 0

    for idx, module in enumerate(modules):
        status = "❄️ FROZEN" if idx in frozen_indices else "✅ TRAINABLE"
        print(f" - Layer {idx:<2}: {module.__class__.__name__:<15} → {status}")

        if idx in frozen_indices:
            for param in module.parameters():
                frozen_param_count += param.numel()

    print(f"\n❄️ Frozen Layer Indices: {frozen_indices}")
    print(f"📦 Total Frozen Parameters: {frozen_param_count:,}")

# 🚀 Custom YOLOv8 training function with configurable freezing and hyperparameters
def train_yolo_custom_freeze(
    data_yaml: str,
    experiment_name: str,
    test_image: str = None,
    freeze_level: int = 0,
    pretrained_weights: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    lr: float = 0.001,
    optimizer: str = "SGD",
    verbose: bool = True
):
    # Suppress excessive logging
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    print(f"\n🚀 Starting training: {experiment_name}")
    print(f"🔒 Custom Freeze Level: {freeze_level} | Optimizer: {optimizer} | LR: {lr} | Batch: {batch} | ImgSize: {imgsz}")

    # Load pre-trained YOLOv8 model
    model = YOLO(pretrained_weights)

    # Print layer freeze info
    print_frozen_layers(model, freeze_level)

    # Start training with given parameters
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=experiment_name,
        optimizer=optimizer,
        lr0=lr,
        freeze=freeze_level,
        verbose=False
    )

    print(f"\n✅ Training completed: {experiment_name}")
    draw_loss_graph(experiment_name)

    # If a test image is provided, run inference and show result
    if test_image and os.path.exists(test_image):
        print(f"\n🧪 Predicting on test image: {test_image}")
        model.predict(source=test_image, save=True, name=f"predict_{experiment_name}", imgsz=imgsz)

        pred_path = f"runs/predict_{experiment_name}/{os.path.basename(test_image)}"
        if os.path.exists(pred_path):
            img = Image.open(pred_path)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Predicted – {experiment_name}")
            plt.show()
        else:
            print("❌ Prediction image could not be loaded.")
    else:
        print("ℹ️ No test image provided or path invalid.")

# %%
# 🧊 Alternative function to print layer freeze status (not used in main flow)
def print_frozen_status(model):
    print("\n📌 Layer Freeze Status (Before Training):")
    for idx, module in enumerate(model.model.model):
        frozen = all(not p.requires_grad for p in module.parameters() if p.requires_grad is not None)
        status = "❄️ FROZEN" if frozen else "✅ Trainable"
        print(f" - Layer {idx:<2}: {module.__class__.__name__:<15} → {status}")

# %% [markdown]
# 🧠 Custom YOLOv8 Training Utility
# 
# This code defines a custom training function `train_yolo_custom_freeze()` to train YOLOv8 models with fine-grained control over training parameters and layer freezing:
# 
# - `freeze_level` controls how many layers (from the start) should be frozen during training.
# - Training settings like optimizer, learning rate, batch size, and image size can be customized.
# - After training, a loss visualization plot is generated for both training and validation losses.
# - If a `test_image` is provided, the model runs inference on it and displays the predicted result.
# - The `print_frozen_layers()` function shows which model layers are frozen before training.
# - The `draw_loss_graph()` function visualizes loss curves from the `results.csv` generated by Ultralytics.
# 
# This utility provides a modular and configurable interface for experimenting with different freeze levels and hyperparameters.

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 📷 Display side-by-side predictions from multiple experiments on the same test image
def plot_experiment_predictions(exp_names, test_img_name, predict_base="runs/detect"):
    plt.figure(figsize=(16, 12))  # Create a large figure

    for i, name in enumerate(exp_names):
        pred_path = f"{predict_base}/predict_{name}/{test_img_name}"  # Construct path to predicted image
        try:
            img = mpimg.imread(pred_path)  # Load the predicted image
            plt.subplot(3, 3, i + 1)  # Arrange in 3x3 grid (supports up to 9 experiments)
            plt.imshow(img)
            plt.title(name)  # Use experiment name as the title
            plt.axis("off")  # Hide axes for better visualization
        except FileNotFoundError:
            print(f"⚠️ Not found: {pred_path}")  # Handle missing images gracefully

    plt.tight_layout()  # Optimize spacing
    plt.show()  # Display the figure

# %% [markdown]
# 🖼️ Prediction Comparison Plot
# 
# This function visualizes the predicted results from multiple YOLOv8 experiments on the same test image:
# 
# - It takes a list of experiment names and the test image filename.
# - For each experiment, it tries to load and display the prediction image from its corresponding folder.
# - Images are arranged in a grid layout (up to 9 experiments).
# - If a prediction is missing, a warning is printed.
# 
# This allows quick visual comparison of how different training configurations perform on the same sample.

# %%
!rm /content/cars_yolo_dataset/labels/train.cache
# Cleaning caches of train dataset.

# %% [markdown]
# ### 🔬 Training Experiments Overview
# 
# This section contains all training experiments carried out under four different freeze configurations. Each configuration focuses on a different set of frozen layers in the YOLOv8 architecture:
# 
# - **Freeze the first 5 blocks:** Includes the stem and the first two C2f blocks. [5 pts]
# - **Freeze the first 10 blocks:** Includes the entire backbone and the SPPF (Spatial Pyramid Pooling - Fast) layer. [5 pts]
# - **Freeze the first 21 blocks:** All layers are frozen except the Detection Head. [5 pts]
# - **No freezing (Full training):** The entire YOLOv8 model is trainable. [5 pts]
# 
# For each freeze setting, **8 distinct models** were trained by varying one of the following hyperparameters:
# 
# - Learning Rate: `0.001` vs `0.0001`
# - Batch Size: `16` vs `64`
# - Optimizer: `SGD` vs `Adam`
# - Image Size: `416` vs `640`
# 
# In total, **32 models** were trained across these 4 freeze levels, allowing comparison of how different configurations and learning setups affect performance.

# %%
# Freeze first 5 layers: stem + 2 C2f blocks
# Each configuration varies one hyperparameter while keeping others fixed
# - 2 different learning rates: 0.001, 0.0001
# - 2 different batch sizes: 16, 64
# - 2 different optimizers: Adam, SGD
# - 2 different image sizes: 416, 640
# → Total of 8 experiments under freeze_level=5

# 🔁 Define experiment configurations for freeze level 5
experiments1 = [
    {"freeze_level": 5, "experiment_name": "frz5_lr001", "lr": 0.001},      # Learning rate = 0.001
    {"freeze_level": 5, "experiment_name": "frz5_lr0001", "lr": 0.0001},    # Learning rate = 0.0001
    {"freeze_level": 5, "experiment_name": "frz5_batch32", "batch": 16},    # Batch size = 16
    {"freeze_level": 5, "experiment_name": "frz5_batch64", "batch": 64},    # Batch size = 64
    {"freeze_level": 5, "experiment_name": "frz5_sgd", "optimizer": "SGD"}, # Optimizer = SGD
    {"freeze_level": 5, "experiment_name": "frz5_adam", "optimizer": "Adam"},# Optimizer = Adam
    {"freeze_level": 5, "experiment_name": "frz5_imgsz416", "imgsz": 416},  # Image size = 416
    {"freeze_level": 5, "experiment_name": "frz5_imgsz640", "imgsz": 640},  # Image size = 640
]

# 🧩 Shared training parameters (used in all experiments)
common_params = {
    "data_yaml": "cars_yolo_dataset/data.yaml",                  # Path to dataset YAML
    "epochs": 30,                                                # Number of training epochs
    "pretrained_weights": "yolov8n.pt",                          # Pretrained YOLOv8 weights
    "test_image": "cars_yolo_dataset/images/test/20160331_NTU_00017.png"  # Inference test image
}

# 🔁 Run all training experiments sequentially
for exp in experiments1:
    train_yolo_custom_freeze(**common_params, **exp)

# %%
# 🖼️ Name of the test image used for visual comparison
test_img_name = "20160331_NTU_00017.jpg"

# 🔢 List of experiment names (corresponding to freeze level 5 models)
exp_names1 = [
    "frz5_lr001",     # Experiment with learning rate 0.001
    "frz5_lr0001",    # Experiment with learning rate 0.0001
    "frz5_batch32",   # Experiment with batch size 16
    "frz5_batch64",   # Experiment with batch size 64
    "frz5_sgd",       # Experiment using SGD optimizer
    "frz5_adam",      # Experiment using Adam optimizer
    "frz5_imgsz416",  # Experiment with image size 416
    "frz5_imgsz640"   # Experiment with image size 640
]

# 📊 Visualize predictions from all experiments on the same test image
plot_experiment_predictions(exp_names1, test_img_name)

# %%
# 📦 Shared training parameters
common_params = {
    "data_yaml": "cars_yolo_dataset/data.yaml",
    "epochs": 30,
    "pretrained_weights": "yolov8n.pt",
    "test_image": "cars_yolo_dataset/images/test/20160331_NTU_00017.png"
}

# 🔁 Training Experiments – Freeze Level 10
# Freezes all backbone layers and SPPF (first 10 layers)
experiments2 = [
    {"freeze_level": 10, "experiment_name": "frz10_lr001", "lr": 0.001},
    {"freeze_level": 10, "experiment_name": "frz10_lr0001", "lr": 0.0001},
    {"freeze_level": 10, "experiment_name": "frz10_batch32", "batch": 16},
    {"freeze_level": 10, "experiment_name": "frz10_batch64", "batch": 64},
    {"freeze_level": 10, "experiment_name": "frz10_sgd", "optimizer": "SGD"},
    {"freeze_level": 10, "experiment_name": "frz10_adam", "optimizer": "Adam"},
    {"freeze_level": 10, "experiment_name": "frz10_imgsz416", "imgsz": 416},
    {"freeze_level": 10, "experiment_name": "frz10_imgsz640", "imgsz": 640},
]

# 🚀 Run each training experiment
for exp in experiments2:
    train_yolo_custom_freeze(**common_params, **exp)

# %%
# 📸 Visualize predictions for all freeze level 10 experiments
exp_names2 = [
    "frz10_lr001", "frz10_lr0001", "frz10_batch32", "frz10_batch64",
    "frz10_sgd", "frz10_adam", "frz10_imgsz416", "frz10_imgsz640"
]
plot_experiment_predictions(exp_names2, test_img_name)

# %%
# 📦 Shared training parameters (redefined for clarity)
common_params = {
    "data_yaml": "cars_yolo_dataset/data.yaml",
    "epochs": 30,
    "pretrained_weights": "yolov8n.pt",
    "test_image": "cars_yolo_dataset/images/test/20160331_NTU_00017.png"
}

# 🔁 Training Experiments – Freeze Level 21
# Freezes all layers except the detection head
experiments3 = [
    {"freeze_level": 21, "experiment_name": "frz21_lr001", "lr": 0.001},
    {"freeze_level": 21, "experiment_name": "frz21_lr0001", "lr": 0.0001},
    {"freeze_level": 21, "experiment_name": "frz21_batch32", "batch": 16},
    {"freeze_level": 21, "experiment_name": "frz21_batch64", "batch": 64},
    {"freeze_level": 21, "experiment_name": "frz21_sgd", "optimizer": "SGD"},
    {"freeze_level": 21, "experiment_name": "frz21_adam", "optimizer": "Adam"},
    {"freeze_level": 21, "experiment_name": "frz21_imgsz416", "imgsz": 416},
    {"freeze_level": 21, "experiment_name": "frz21_imgsz640", "imgsz": 640},
]

# 🚀 Run all experiments under freeze level 21
for exp in experiments3:
    train_yolo_custom_freeze(**common_params, **exp)

# %%
# 📸 Visualize predictions for freeze level 21
exp_names3 = [
    "frz21_lr001", "frz21_lr0001", "frz21_batch32", "frz21_batch64",
    "frz21_sgd", "frz21_adam", "frz21_imgsz416", "frz21_imgsz640"
]
plot_experiment_predictions(exp_names3, test_img_name)

# %%
# 📦 Shared training parameters (again declared for full isolation)
common_params = {
    "data_yaml": "cars_yolo_dataset/data.yaml",
    "epochs": 30,
    "pretrained_weights": "yolov8n.pt",
    "test_image": "cars_yolo_dataset/images/test/20160331_NTU_00017.png"
}

# 🧠 Training experiments — No freezing (full network is trainable)
experiments4 = [
    {"freeze_level": 0, "experiment_name": "frz0_lr001", "lr": 0.001},
    {"freeze_level": 0, "experiment_name": "frz0_lr0001", "lr": 0.0001},
    {"freeze_level": 0, "experiment_name": "frz0_batch32", "batch": 16},
    {"freeze_level": 0, "experiment_name": "frz0_batch64", "batch": 64},
    {"freeze_level": 0, "experiment_name": "frz0_sgd", "optimizer": "SGD"},
    {"freeze_level": 0, "experiment_name": "frz0_adam", "optimizer": "Adam"},
    {"freeze_level": 0, "experiment_name": "frz0_imgsz416", "imgsz": 416},
    {"freeze_level": 0, "experiment_name": "frz0_imgsz640", "imgsz": 640},
]

# 🚀 Run each experiment
for exp in experiments4:
    train_yolo_custom_freeze(**common_params, **exp)

# %%
# 📊 Visual comparison of predictions
exp_names4 = [
    "frz0_lr001", "frz0_lr0001", "frz0_batch32", "frz0_batch64",
    "frz0_sgd", "frz0_adam", "frz0_imgsz416", "frz0_imgsz640"
]

plot_experiment_predictions(exp_names4, test_img_name)

# %% [markdown]
# ### 📊 Evaluation & Qualitative Result Functions:
# 
# This section contains functions used for evaluating the trained YOLOv8 models in both quantitative and qualitative ways:
# 
# - `show_success_and_missed_examples`: Displays one successful and one missed detection based on whether the model found any objects.
# - `show_class_count_based_examples`: Compares prediction count vs ground truth count for class 0 (cars) and visualizes:
#   - A fully detected example
#   - A partially missed example
#   - A fully missed example
# - `evaluate_detector_metrics`: Computes detection metrics such as:
#   - Exact Match Accuracy (based on object count equality)
#   - Mean Squared Error (MSE) of predicted object counts
#   - Precision and Recall based on class 0 object count comparison
# 
# These tools are helpful for both debugging and comparing how well different models generalize.

# %%
# ✅ Show one successful and one missed detection for a model
def show_success_and_missed_examples(model_path, conf=0.3):
    model = YOLO(model_path)
    model_name = Path(model_path).parts[-3]  # Extract experiment name

    test_images = list(Path("cars_yolo_dataset/images/test").glob("*.png"))
    missed = []
    successful = []

    # Run prediction on each test image
    for img_path in test_images:
        results = model.predict(source=str(img_path), conf=conf, save=False, verbose=False)
        boxes = results[0].boxes
        num_preds = len(boxes)

        if num_preds == 0:
            missed.append((img_path, results[0]))
        else:
            successful.append((img_path, results[0]))

    # 🔹 Show a successful detection
    if successful:
        img_path, result = successful[0]
        print(f"✅ Showing successful prediction: {img_path.name} from {model_name}")
        img = result.plot()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"[{model_name}] {img_path.name} – Found {len(result.boxes)} boxes ✅")
        plt.show()
    else:
        print(f"⚠️ No successful detections found for {model_name}")

    # 🔸 Show a missed detection
    if missed:
        img_path, result = missed[0]
        print(f"❌ Showing missed prediction: {img_path.name} from {model_name}")
        img = result.orig_img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"[{model_name}] {img_path.name} – No boxes ❌")
        plt.show()
    else:
        print(f"🎯 No missed detections – model found objects in all test images for {model_name}")

# %%
# 📏 Compare prediction counts with ground truth labels and show examples
def show_class_count_based_examples(model_path, conf=0.3):
    model = YOLO(model_path)
    model_name = Path(model_path).parts[-3]

    test_images = list(Path("cars_yolo_dataset/images/test").glob("*.png"))
    label_dir = Path("cars_yolo_dataset/labels/test")

    fully_detected = None
    partially_detected = None
    fully_missed = None
    missed_count = 0  # Number of missed car objects

    # Loop through each test image
    for img_path in test_images:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        # Count ground truth class 0 objects
        with open(label_path, "r") as f:
            gt_class0_count = sum(1 for line in f if line.strip().startswith("0"))

        if gt_class0_count == 0:
            continue

        # Run prediction
        results = model.predict(source=str(img_path), conf=conf, save=False, verbose=False)
        pred_class0_count = sum(1 for b in results[0].boxes.cls if int(b) == 0)

        # Classify the result
        if pred_class0_count == 0 and not fully_missed:
            fully_missed = (img_path, results[0])
        elif pred_class0_count == gt_class0_count and not fully_detected:
            fully_detected = (img_path, results[0])
        elif pred_class0_count < gt_class0_count and not partially_detected:
            partially_detected = (img_path, results[0])
            missed_count = gt_class0_count - pred_class0_count

        if fully_detected and partially_detected and fully_missed:
            break

    # Utility function for displaying results
    def display(img_path, result, title):
        img = result.plot()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"[{model_name}] {img_path.name} – {title}")
        plt.show()

    if fully_detected:
        print(f"✅ Fully Detected: {fully_detected[0].name}")
        display(*fully_detected, "Predicted correct number of cars ✅")
    else:
        print("❌ No fully detected examples found.")

    if partially_detected:
        print(f"⚠️ Partially Missed: {partially_detected[0].name}")
        print(f"   Ground truth cars: {missed_count + sum(1 for _ in partially_detected[1].boxes.cls if int(_) == 0)}")
        print(f"   Predicted cars: {sum(1 for _ in partially_detected[1].boxes.cls if int(_) == 0)}")
        print(f"   ❗ Missed {missed_count} car(s)")
        display(*partially_detected, f"Missed {missed_count} car(s) ⚠️")
    else:
        print("❌ No partially missed examples found.")

    if fully_missed:
        print(f"❌ Fully Missed: {fully_missed[0].name}")
        display(*fully_missed, "Model predicted no cars ❌")
    else:
        print("🎯 No fully missed examples found.")

# %%
# 📊 Compute quantitative metrics based on class-0 object count
def evaluate_detector_metrics(model_path, conf=0.3):
    model = YOLO(model_path)
    test_images = list(Path("cars_yolo_dataset/images/test").glob("*.png"))
    label_dir = Path("cars_yolo_dataset/labels/test")

    exact_matches = 0
    squared_errors = []
    tp = 0  # true positives
    fp = 0  # false positives
    fn = 0  # false negatives

    for img_path in test_images:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        # Count ground truth cars
        with open(label_path, "r") as f:
            gt_class0 = [line for line in f if line.strip().startswith("0")]
        gt_count = len(gt_class0)

        # Predict
        results = model.predict(source=str(img_path), conf=conf, save=False, verbose=False)
        preds = results[0]

        # Count predicted cars
        pred_class0 = [int(cls) for cls in preds.boxes.cls if int(cls) == 0]
        pred_count = len(pred_class0)

        # Evaluate exact match and accumulate squared error
        if pred_count == gt_count:
            exact_matches += 1
        squared_errors.append((pred_count - gt_count) ** 2)

        # Count-based approximation for TP, FP, FN
        tp += min(pred_count, gt_count)
        fp += max(pred_count - gt_count, 0)
        fn += max(gt_count - pred_count, 0)

    # Final metrics
    total = len(test_images)
    exact_match_accuracy = (exact_matches / total) * 100 if total > 0 else 0
    mse = np.mean(squared_errors) if squared_errors else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "Exact Match Accuracy (%)": round(exact_match_accuracy, 2),
        "MSE": round(mse, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3)
    }

# %% [markdown]
# ### 🔵 Freeze First 5 Layers (Stem + First 2 C2f Blocks)
# 
# The following visualizations display detection outcomes from 8 different YOLOv8 models trained with the first 5 layers frozen (including the stem and the first two C2f blocks). Each model uses a distinct hyperparameter configuration — varying either the learning rate, batch size, optimizer, or image size.
# 
# For each model, we show:
# - ✅ One **successful prediction**, where the model detects at least one object.
# - ❌ One **missed prediction**, where the model fails to detect any objects.
# 
# These examples provide visual insights into how different training setups affect detection performance under partial layer freezing.

# %%
# 📌 NOTE:
# The function `show_success_and_missed_examples` was tested only for a few selected models
# (one from each freeze level) to verify its behavior. It is not the primary visualization tool.
# The main visualization function used in this evaluation is `show_class_count_based_examples`.

# 🔍 Qualitative evaluation: one successful and one missed example per freeze level
model_names = [
    "frz5_lr001",  # Freeze Level 5
    "frz10_lr001", # Freeze Level 10
    "frz21_lr001", # Freeze Level 21
    "frz0_lr001"   # No freezing
]

for name in model_names:
    print("\n" + "="*60)
    print(f"📌 Results for model: {name}")
    print("="*60)
    show_success_and_missed_examples(f"runs/detect/{name}/weights/best.pt")

# %%
# 📊 Main qualitative visualization: class count–based detection analysis
# For each experiment, the function shows:
# - A fully detected sample (predicted cars = ground truth)
# - A partially missed sample (predicted cars < ground truth)
# - A fully missed sample (no cars detected)

model_names = [
    "frz5_lr001", "frz5_lr0001",
    "frz5_batch32", "frz5_batch64",
    "frz5_sgd", "frz5_adam",
    "frz5_imgsz416", "frz5_imgsz640"
]

for name in model_names:
    print("\n" + "="*60)
    print(f"📌 Results for model: {name}")
    print("="*60)
    show_class_count_based_examples(f"runs/detect/{name}/weights/best.pt")

# %% [markdown]
# ### 🟠 Freeze First 10 Layers (Backbone + SPPF Frozen)
# 
# These results are from 8 models trained with the first 10 layers frozen, including the entire backbone and the Spatial Pyramid Pooling - Fast (SPPF) module.
# 
# The objective is to evaluate how well the detection head can adapt when the feature extractor is kept fixed. Each model was trained with a different hyperparameter configuration.
# 
# For each case, we display:
# - ✅ One successful detection (at least one object correctly identified)
# - ❌ One failed detection (no objects detected)

# %%
# 📊 Main qualitative visualization: class count–based detection analysis
# For each experiment, the function shows:
# - A fully detected sample (predicted cars = ground truth)
# - A partially missed sample (predicted cars < ground truth)
# - A fully missed sample (no cars detected)

model_names = [
    "frz10_lr001", "frz10_lr0001",
    "frz10_batch32", "frz10_batch64",
    "frz10_sgd", "frz10_adam",
    "frz10_imgsz416", "frz10_imgsz640"
]

for name in model_names:
    print("\n" + "="*60)
    print(f"📌 Results for model: {name}")
    print("="*60)
    show_class_count_based_examples(f"runs/detect/{name}/weights/best.pt")

# %% [markdown]
# ### 🟣 Freeze First 21 Layers (Only Detection Head Trained)
# 
# In this configuration, all layers except the detection head were frozen. Only the final layers responsible for object localization and classification were updated during training.
# 
# The visualizations below show detection results from 8 different models trained with varying hyperparameters. Each model output includes:
# - ✅ One image where all objects were correctly detected
# - ❌ One image where detection failed entirely

# %%
# 📊 Main qualitative visualization: class count–based detection analysis
# For each experiment, the function shows:
# - A fully detected sample (predicted cars = ground truth)
# - A partially missed sample (predicted cars < ground truth)
# - A fully missed sample (no cars detected)

model_names = [
    "frz21_lr001", "frz21_lr0001",
    "frz21_batch32", "frz21_batch64",
    "frz21_sgd", "frz21_adam",
    "frz21_imgsz416", "frz21_imgsz640"
]

for name in model_names:
    print("\n" + "="*60)
    print(f"📌 Results for model: {name}")
    print("="*60)
    show_class_count_based_examples(f"runs/detect/{name}/weights/best.pt")

# %% [markdown]
# ### 🟢 Full Model Training (No Layers Frozen)
# 
# This section shows the results from 8 fully trainable YOLOv8 models, where no layers were frozen during training. This allows the network to perform full fine-tuning across all components.
# 
# Each model was trained with a different hyperparameter setting. The visual results demonstrate how well the models learned to detect cars under different training conditions:
# - ✅ One image with successful detection
# - ❌ One image with missed detection

# %%
# 📊 Main qualitative visualization: class count–based detection analysis
# For each experiment, the function shows:
# - A fully detected sample (predicted cars = ground truth)
# - A partially missed sample (predicted cars < ground truth)
# - A fully missed sample (no cars detected)

model_names = [
    "frz0_lr001", "frz0_lr0001",
    "frz0_batch32", "frz0_batch64",
    "frz0_sgd", "frz0_adam",
    "frz0_imgsz416", "frz0_imgsz640"
]

for name in model_names:
    print("\n" + "="*60)
    print(f"📌 Results for model: {name}")
    print("="*60)
    show_class_count_based_examples(f"runs/detect/{name}/weights/best.pt")

# %% [markdown]
# ### 📊 Evaluation of Best Models Across Training Configurations
# 
# In this section, we quantitatively evaluate and compare the performance of the best YOLOv8 models trained under various hyperparameter configurations. Each configuration was tested using four different freeze strategies:
# 
# - **Freeze-5**
# - **Freeze-10**
# - **Freeze-21**
# - **Full-Train**
# 
# The evaluation is conducted on the full test set using the following metrics:
# 
# - **Exact Match Accuracy (%):** How often the predicted number of objects exactly matches ground truth.
# - **Mean Squared Error (MSE):** Measures prediction deviation from ground truth count.
# - **Precision:** Percentage of predicted cars that are true positives.
# - **Recall:** Percentage of actual cars that are correctly predicted.
# 
# We performed 8 different hyperparameter experiments to assess the effect of:
# 
# - Learning rates: `0.001` and `0.0001`
# - Batch sizes: `16` and `64`
# - Optimizers: `SGD` and `Adam`
# - Image sizes: `416` and `640`
# 
# Each result block below presents the detection performance of all four freeze strategies under a single hyperparameter setting.
# 
# > 📌 This analysis aims to reveal which combination of training configuration and freeze strategy produces the most robust and accurate car detection model.

# %% [markdown]
# 🗃️ Structure and Purpose of `all_results`
# 
# To keep track of all evaluation outcomes across different configurations, we use a nested dictionary named `all_results`.
# 
# The structure of `all_results` is as follows:
# 
# ```python
# all_results = {
#     "lr001": {
#         "Freeze-5": { ...metrics... },
#         "Freeze-10": { ...metrics... },
#         ...
#     },
#     "batch64": {
#         "Freeze-5": { ...metrics... },
#         ...
#     },
#     ...
# 

# %%
all_results = {}

# %% [markdown]
# 🧪 Logging Model Evaluation Results by Configuration
# 
# Each code block below corresponds to a different training configuration—such as learning rate (`lr001`, `lr0001`), batch size (`batch32`, `batch64`), optimizer (`adam`, `sgd`), or image size (`imgsz416`, `imgsz640`).
# 
# For every configuration, the performance of four freeze strategies (`Freeze-5`, `Freeze-10`, `Freeze-21`, `Full-Train`) is evaluated using the `evaluate_detector_metrics()` function. These metrics are stored in a structured Python dictionary named `all_results`.
# 
# The results are grouped under a key that reflects the current configuration, allowing easy access and comparison across different experimental setups. For example:
# 
# ```python
# all_results["batch64"]["Freeze-10"] → precision, recall, MSE, accuracy, etc.

# %%
model_group = {
    "Freeze-5": "frz5_lr001",
    "Freeze-10": "frz10_lr001",
    "Freeze-21": "frz21_lr001",
    "Full-Train": "frz0_lr001"
}

all_results["lr001"] = {}

for label, name in model_group.items():
    print(f"\n🔍 Evaluating {label} → {name}")
    metrics = evaluate_detector_metrics(f"runs/detect/{name}/weights/best.pt")
    all_results["lr001"][label] = metrics

    print("📊 Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

# %%
model_group = {
    "Freeze-5": "frz5_lr0001",
    "Freeze-10": "frz10_lr0001",
    "Freeze-21": "frz21_lr0001",
    "Full-Train": "frz0_lr0001"
}

all_results["lr0001"] = {}

for label, name in model_group.items():
    print(f"\n🔍 Evaluating {label} → {name}")
    metrics = evaluate_detector_metrics(f"runs/detect/{name}/weights/best.pt")
    all_results["lr0001"][label] = metrics

    print("📊 Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

# %%
model_group = {
    "Freeze-5": "frz5_batch32",
    "Freeze-10": "frz10_batch32",
    "Freeze-21": "frz21_batch32",
    "Full-Train": "frz0_batch32"
}

all_results["batch32"] = {}

for label, name in model_group.items():
    print(f"\n🔍 Evaluating {label} → {name}")
    metrics = evaluate_detector_metrics(f"runs/detect/{name}/weights/best.pt")
    all_results["batch32"][label] = metrics

    print("📊 Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

# %%
model_group = {
    "Freeze-5": "frz5_batch64",
    "Freeze-10": "frz10_batch64",
    "Freeze-21": "frz21_batch64",
    "Full-Train": "frz0_batch64"
}

all_results["batch64"] = {}

for label, name in model_group.items():
    print(f"\n🔍 Evaluating {label} → {name}")
    metrics = evaluate_detector_metrics(f"runs/detect/{name}/weights/best.pt")
    all_results["batch64"][label] = metrics

    print("📊 Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

# %%
model_group = {
    "Freeze-5": "frz5_sgd",
    "Freeze-10": "frz10_sgd",
    "Freeze-21": "frz21_sgd",
    "Full-Train": "frz0_sgd"
}

all_results["sgd"] = {}

for label, name in model_group.items():
    print(f"\n🔍 Evaluating {label} → {name}")
    metrics = evaluate_detector_metrics(f"runs/detect/{name}/weights/best.pt")
    all_results["sgd"][label] = metrics

    print("📊 Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

# %%
model_group = {
    "Freeze-5": "frz5_adam",
    "Freeze-10": "frz10_adam",
    "Freeze-21": "frz21_adam",
    "Full-Train": "frz0_adam"
}

all_results["adam"] = {}

for label, name in model_group.items():
    print(f"\n🔍 Evaluating {label} → {name}")
    metrics = evaluate_detector_metrics(f"runs/detect/{name}/weights/best.pt")
    all_results["adam"][label] = metrics

    print("📊 Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

# %%
model_group = {
    "Freeze-5": "frz5_imgsz416",
    "Freeze-10": "frz10_imgsz416",
    "Freeze-21": "frz21_imgsz416",
    "Full-Train": "frz0_imgsz416"
}

all_results["imgsz416"] = {}

for label, name in model_group.items():
    print(f"\n🔍 Evaluating {label} → {name}")
    metrics = evaluate_detector_metrics(f"runs/detect/{name}/weights/best.pt")
    all_results["imgsz416"][label] = metrics

    print("📊 Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

# %%
model_group = {
    "Freeze-5": "frz5_imgsz640",
    "Freeze-10": "frz10_imgsz640",
    "Freeze-21": "frz21_imgsz640",
    "Full-Train": "frz0_imgsz640"
}

all_results["imgsz640"] = {}

for label, name in model_group.items():
    print(f"\n🔍 Evaluating {label} → {name}")
    metrics = evaluate_detector_metrics(f"runs/detect/{name}/weights/best.pt")
    all_results["imgsz640"][label] = metrics

    print("📊 Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

# %% [markdown]
# ## **MSE, Exact Match Accuracy, Precision and Recall Plots for 4 different freezing strategies:**

# %% [markdown]
# 📈 Plotting Metric Trends Across Configurations
# 
# The `plot_param_comparison_grid(all_results)` function visualizes how key metrics (MSE, Accuracy, Precision, Recall) change across different training configurations, grouped by each freeze strategy. Each subplot shows one metric under one freeze setting, with x-axis labels representing hyperparameter scenarios (e.g., `lr001`, `batch64`, etc.).
# 
# > 🔍 This compact view helps identify stable or sensitive behaviors across setups.

# %%
import matplotlib.pyplot as plt

def plot_param_comparison_grid(all_results):
    freeze_types = ["Freeze-5", "Freeze-10", "Freeze-21", "Full-Train"]
    metrics = ["MSE", "Exact Match Accuracy (%)", "Precision", "Recall"]

    fig, axes = plt.subplots(len(freeze_types), len(metrics), figsize=(20, 16))
    fig.suptitle("Metric Comparison Across Parameters (Grouped by Freeze Strategy)", fontsize=18)

    for row_idx, freeze_type in enumerate(freeze_types):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            param_settings = []
            values = []

            for param, freeze_dict in all_results.items():
                if freeze_type in freeze_dict:
                    param_settings.append(param)
                    values.append(freeze_dict[freeze_type][metric])

            ax.plot(param_settings, values, marker="o", linestyle="--", color="tab:blue")
            ax.set_title(f"{freeze_type} – {metric}", fontsize=10)
            ax.set_xticks(range(len(param_settings)))
            ax.set_xticklabels(param_settings, rotation=45)
            ax.grid(True)

            if col_idx == 0:
                ax.set_ylabel("Value", fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %% [markdown]
# 🔍 Freeze Strategy Comparison for a Selected Metric
# 
# The `plot_freeze_comparison(all_results, metric_name)` function plots how a chosen performance metric (e.g., MSE, Accuracy) varies across the four freeze strategies for each training configuration.
# 
# Each line in the chart corresponds to a parameter setting (like `lr001`, `batch64`, etc.), and compares the effect of different freezing levels on the selected metric.
# 
# > 📌 This plot helps identify which freeze level is most effective under each configuration.

# %%
import matplotlib.pyplot as plt

def plot_freeze_comparison(all_results, metric_name):
    plt.figure(figsize=(12, 6))  # Create a wide figure for clarity

    for param_setting, freeze_metrics in all_results.items():
        metric_values = []
        freeze_order = ["Freeze-5", "Freeze-10", "Freeze-21", "Full-Train"]  # Consistent freeze order

        # Collect the metric values for each freeze strategy
        for freeze_type in freeze_order:
            if freeze_type in freeze_metrics:
                metric_values.append(freeze_metrics[freeze_type][metric_name])
            else:
                metric_values.append(None)  # Preserve position even if data is missing

        # Plot metric values for this parameter setting
        plt.plot(freeze_order, metric_values, marker="o", label=param_setting)

    # Configure plot aesthetics
    plt.title(f"{metric_name} – Comparison Between Freezing Strategies")
    plt.xlabel("Freezing Strategy")
    plt.ylabel(metric_name)
    plt.legend(title="Training Param")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# 📊 Parameter Sensitivity Analysis for a Single Freeze Strategy
# 
# The `plot_param_comparison(all_results, metric_name, freeze_target)` function visualizes how a selected metric (e.g., MSE, Accuracy) changes across different training configurations (e.g., `lr001`, `batch64`, etc.) for a **single freeze strategy**.
# 
# This view helps assess the robustness or sensitivity of a particular freeze level under various training setups.
# 
# > 📌 Ideal for spotting stable patterns or inconsistencies tied to specific configurations.

# %%
def plot_param_comparison(all_results, metric_name, freeze_target):
    param_settings = []  # X-axis: parameter setting names (e.g., 'lr001')
    values = []          # Y-axis: metric values (e.g., accuracy, MSE)

    # Loop through all configurations and collect the metric if the freeze type exists
    for param, freeze_dict in all_results.items():
        if freeze_target in freeze_dict:
            param_settings.append(param)
            values.append(freeze_dict[freeze_target][metric_name])

    # Plot the metric across parameter settings
    plt.figure(figsize=(12, 6))
    plt.plot(param_settings, values, marker="o", linestyle="--", color="darkblue")
    plt.title(f"{metric_name} – Across Params for {freeze_target}")
    plt.xlabel("Training Param Setting")
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# 🏆 Top 4 Performing Models by Metric
# 
# The `plot_top4_models_per_metric(all_results, metric)` function identifies the top 4 performing models across **all configurations and freeze strategies** based on a selected evaluation metric.
# 
# - For **MSE**, it selects the 4 models with the **lowest** values.
# - For all other metrics (e.g., Accuracy, Precision), it selects the **highest**.
# 
# The final result is a horizontal bar chart showing the metric values and their corresponding configuration + freeze strategy label.
# 
# > 🧠 This plot is useful for quickly identifying the best combinations regardless of parameter or freeze setting.

# %%
def plot_top4_models_per_metric(all_results, metric):
    entries = []

    # Flatten all metric values with labels into a list
    for param_setting, freeze_dict in all_results.items():
        for freeze_type, metrics_dict in freeze_dict.items():
            value = metrics_dict[metric]
            label = f"{param_setting} | {freeze_type}"
            entries.append((label, value))

    # Select top 4 entries based on sorting direction
    if metric == "MSE":
        top4 = sorted(entries, key=lambda x: x[1])[:4]  # Lowest 4 MSE
    else:
        top4 = sorted(entries, key=lambda x: x[1], reverse=True)[:4]  # Highest 4 for other metrics

    # Unpack for plotting
    labels, values = zip(*top4)

    # Create horizontal bar chart
    plt.figure(figsize=(8, 5))
    plt.barh(labels, values, color="skyblue")
    plt.xlabel(metric)
    plt.title(f"Top 4 Models by {metric}")
    plt.gca().invert_yaxis()  # Best at the top

    # Add value labels to bars
    for i, v in enumerate(values):
        plt.text(v + 0.5, i, f"{v:.2f}", va='center')

    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# 🧩 Top Models from Distinct Parameter Groups
# 
# The `plot_best_models_diverse_groups(all_results, metric)` function selects the **best-performing model from each high-level parameter group** (e.g., `lr001`, `batch64`, etc.) based on a specified metric.
# 
# It then visualizes the **top 4 models** across different groups using a clean vertical bar chart. Only one model per group is allowed, ensuring diversity across training configurations.
# 
# - For **MSE**, lower is better — selects the 4 smallest values.
# - For other metrics (e.g., Accuracy, Recall), higher is better — selects the 4 largest.
# 
# > 🎯 This plot helps highlight which parameter groups consistently produce top results, not just individual outliers.

# %%
import matplotlib.pyplot as plt

def plot_best_models_diverse_groups(all_results, metric):

    best_from_each_group = []

    # Iterate over each high-level parameter group
    for top_key, subdict in all_results.items():
        best_value = None
        best_model = None

        # Within each group, find the best freeze strategy for the given metric
        for subkey, metrics_dict in subdict.items():
            if metric not in metrics_dict:
                continue

            value = metrics_dict[metric]

            # Select based on metric type (min for MSE, max for others)
            if best_value is None or (value < best_value if metric == "MSE" else value > best_value):
                best_value = value
                best_model = (f"{subkey} [{top_key}]", value)

        # Save the best model for this group
        if best_model:
            best_from_each_group.append(best_model)

    # From the best in each group, select top 4 globally
    if metric == "MSE":
        top4 = sorted(best_from_each_group, key=lambda x: x[1])[:4]
    else:
        top4 = sorted(best_from_each_group, key=lambda x: x[1], reverse=True)[:4]

    # Extract labels and metric values for plotting
    labels, values = zip(*top4)

    # Set y-axis range with small padding for readability
    value_min = min(values)
    value_max = max(values)
    value_range = value_max - value_min
    y_min = value_min - value_range * 0.1
    y_max = value_max + value_range * 0.15

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color="mediumseagreen", width=0.5)

    # Add text labels above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * value_range,
                 f"{yval:.2f}", ha='center', va='bottom', fontsize=9)

    # Final plot settings
    plt.ylabel(metric)
    plt.title(f"Top 4 Diverse Models by {metric}", fontsize=12)
    plt.ylim(y_min, y_max)
    plt.xticks(rotation=20)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **All plot results**

# %%
plot_param_comparison_grid(all_results)

# %% [markdown]
# **Freeze strategies comparisons:**

# %%
plot_freeze_comparison(all_results, "MSE")
plot_freeze_comparison(all_results, "Precision")
plot_freeze_comparison(all_results, "Recall")

# %%
metrics = ["MSE", "Exact Match Accuracy (%)", "Precision", "Recall"]

# %%
for metric in metrics:
    print(f"📊 Plotting {metric} for Freeze-5...")
    plot_param_comparison(all_results, metric, "Freeze-5")

# %%
for metric in metrics:
    print(f"📊 Plotting {metric} for Freeze-10...")
    plot_param_comparison(all_results, metric, "Freeze-10")

# %%
for metric in metrics:
    print(f"📊 Plotting {metric} for Freeze-21...")
    plot_param_comparison(all_results, metric, "Freeze-21")

# %%
for metric in metrics:
    print(f"📊 Plotting {metric} for Full-Train...")
    plot_param_comparison(all_results, metric, "Full-Train")

# %% [markdown]
# # **Best Models for every parameter:**

# %%
plot_top4_models_per_metric(all_results, "Exact Match Accuracy (%)")
plot_top4_models_per_metric(all_results, "MSE")
plot_top4_models_per_metric(all_results, "Precision")
plot_top4_models_per_metric(all_results, "Recall")

# %%
plot_best_models_diverse_groups(all_results, "Exact Match Accuracy (%)")
plot_best_models_diverse_groups(all_results, "MSE")
plot_best_models_diverse_groups(all_results, "Precision")
plot_best_models_diverse_groups(all_results, "Recall")

# %% [markdown]
# # **RESULTS EXPLANATION**

# %% [markdown]
# 📊 Combined Model Evaluation Table
# 
# To enable side-by-side comparison of all 32 YOLO model configurations, this script loads the previously saved `all_results.json` file. While the file contents are structured as a nested dictionary organized by configuration and freeze strategy, this structure is flattened into a tabular format for readability.
# 
# The resulting table showcases performance metrics — **Exact Match Accuracy**, **MSE**, **Precision**, and **Recall** — for every configuration. Each row represents a unique combination of training strategy and hyperparameter setting.

# %%
import json
import pandas as pd

# This script loads the full evaluation results that were previously saved in all_results.json.
# It retrieves the performance metrics (accuracy, MSE, precision, recall) for all 32 model configurations
# so that we can sort, analyze, and visualize them as needed.

# Load the all_results JSON file
with open("all_results.json", "r") as f:
    all_results = json.load(f)

# Define desired order for freeze strategies and configurations
freeze_order = ["Freeze-5", "Freeze-10", "Freeze-21", "Full-Train"]
config_order = ["lr001", "lr0001", "batch32", "batch64", "sgd", "adam", "imgsz416", "imgsz640"]

# Flatten the nested dictionary into a list of records
records = []
for freeze in freeze_order:
    for config in config_order:
        if config in all_results and freeze in all_results[config]:
            metrics = all_results[config][freeze]
            records.append({
                "Freeze Strategy": freeze,
                "Configuration": config,
                "Accuracy (%)": metrics["Exact Match Accuracy (%)"],
                "MSE": metrics["MSE"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"]
            })

# Create a DataFrame from the records
df_ordered = pd.DataFrame(records)

# Print the table in markdown format (easy to paste into documentation)
print(df_ordered.to_markdown(index=False))

# %% [markdown]
# ## 📊 RESULTS ANALYSIS & CONCLUSION
# 
# ---
# > ⚠️ **Note on Batch Size Naming:**  
# During training, model names were mistakenly saved using `batch32` and `batch64` labels.  
# However, the actual batch sizes used were **`batch16` and `batch64`**.  
# Therefore, in the *Results* section and all related analysis, we refer to the correct setting as **`batch16`** wherever applicable (e.g., `batch32` → `batch16`).
# 
# ---
# 
# ### 🔍 1. Freeze Strategy Impact
# 
# The effect of **layer freezing** strategies was consistent and highly informative across all training configurations. Here's how each freeze level behaved:
# 
# ---
# 
# #### 🔓 Full-Train (Freeze-0)
# - This strategy trains the **entire model** end-to-end, leveraging the full capacity of the backbone and detection head.
# - It generally produced **strong results** across most configurations.
#   - For example, in `adam` + `imgsz640`, it achieved 40.0% accuracy and MSE of 6.21.
# - However, it requires **maximum training time and compute**, and in some cases, **Freeze-5 matched or outperformed it** with fewer trainable parameters.
# 
# > 📌 **Takeaway:** Full-Train provides strong results but is computationally expensive.
# 
# ---
# 
# #### ❄️ Freeze-5
# - Only the early layers (stem + first 2 blocks) are frozen.
# - This strategy consistently achieved performance **on par with or better than Full-Train** in nearly every configuration.
#   - In `adam`, it reached the **best overall result** among initial experiments: 41.0% accuracy and MSE of 5.65 (batch size: 16).
# - It is a **great balance between efficiency and performance**.
# 
# > ✅ **Best general-purpose option** when you want to reduce training cost without sacrificing accuracy.
# 
# ---
# 
# #### ❄️ Freeze-10
# - This freezes the entire backbone and the SPPF module.
# - Performance dropped slightly compared to Freeze-5 and Full-Train but remained competitive, especially when paired with `adam` and larger images.
# - It’s a reasonable option when **more freezing is needed** (e.g., for speed), but it’s **not optimal**.
# 
# > ⚖️ **Use only if training time must be reduced further** at some performance cost.
# 
# ---
# 
# #### ❄️ Freeze-21
# - All layers except the detection head are frozen.
# - This consistently led to the **worst performance**, with extremely low accuracy (often <10%) and very high MSE (up to 731.77).
# - The model lacked capacity to adapt to the counting task under this setting.
# 
# > ❌ **Avoid this configuration** unless you are doing inference-only fine-tuning with very limited data.
# 
# ---
# 
# > ✅ **Overall Summary:**  
# - **Freeze-5** is the sweet spot — it offers near-maximum performance with lower training cost.
# - **Full-Train** is still solid, but only marginally better (or sometimes worse) than Freeze-5.
# - **Freeze-10** is acceptable but not ideal.
# - **Freeze-21** severely limits learning and should be avoided.
# 
# ---
# 
# ### ⚙️ 2. Configuration-Level Comparisons
# 
# The impact of individual training configurations on model performance is summarized below. Each setting was tested across all freeze strategies, allowing for clear comparisons.
# 
# ---
# 
# #### 🔸 Learning Rate: `lr001` vs. `lr0001`
# 
# - **`lr001` consistently outperformed `lr0001`** across all strategies.
#   - Example: Full-Train  
#     - `lr001`: **30.5% accuracy**, MSE = 16.35  
#     - `lr0001`: **14.5% accuracy**, MSE = 206.36
# - Low learning rate (`lr0001`) led to **underfitting**, especially when freezing was applied.
# 
# > ✅ **Conclusion:** `lr001` (0.001) is clearly more effective for this task.
# 
# ---
# 
# #### 🔸 Batch Size: `batch16` vs. `batch64` *(Originally saved as batch32 and batch64)*
# 
# - When **training the full network**, `batch16` and `batch64` performed similarly.
# - However, under **higher freeze levels** like Freeze-10 or Freeze-21, `batch64` showed **slightly better stability** and lower MSE.
#   - Likely due to smoother gradient estimation with larger batches.
# - Based on this insight, a new model was trained with `batch64` using the same top-performing parameters (Freeze-5 + Adam + lr001 + imgsz640), and it yielded even better results.
# 
# > ⚠️ **Note:** What was labeled as `batch32` in model names actually corresponds to **`batch16`** used during training.
# 
# > ✅ **Conclusion:** Prefer `batch64` in moderate/heavy freezing. It offers better gradient stability and final performance.
# 
# ---
# 
# #### 🔸 Optimizer: `SGD` vs. `Adam`
# 
# - **Adam outperformed SGD in all cases**, particularly with partial freezing.
#   - Example: Freeze-5  
#     - Adam: **41.0% accuracy**, MSE = **5.65**  
#     - SGD: **28.5% accuracy**, MSE = 24.83
# - Adam's adaptive learning rates provided **more stable convergence**, especially when the network's capacity was partially frozen.
# 
# > ✅ **Conclusion:** Use **Adam** for more consistent and superior results.
# 
# ---
# 
# #### 🔸 Image Size: `imgsz416` vs. `imgsz640`
# 
# - `imgsz640` clearly outperformed `imgsz416` across all metrics.
#   - Example: Full-Train  
#     - `imgsz640`: **30.5% accuracy**, MSE = 16.35  
#     - `imgsz416`: **18.5% accuracy**, MSE = 117.14
# - The higher resolution provided **better localization** and more accurate object counting.
# 
# > ✅ **Conclusion:** Always prefer `imgsz640` unless computational limits demand smaller inputs.
# 
# ---
# 
# ### 🧩 Overall Takeaway
# 
# | Parameter      | Recommended Value | Reason                                |
# |----------------|-------------------|----------------------------------------|
# | Learning Rate  | `lr001`           | Prevents underfitting, better accuracy |
# | Batch Size     | `batch64`         | More stable with frozen layers         |
# | Optimizer      | `Adam`            | Stronger convergence, lower MSE        |
# | Image Size     | `imgsz640`        | Better detection and counting          |
# 
# > ✅ Combine these with `Freeze-5` for optimal results across accuracy, precision, and training efficiency.

# %%
train_yolo_custom_freeze(
    data_yaml="cars_yolo_dataset/data.yaml",
    experiment_name="frz5_adam_final",
    test_image="cars_yolo_dataset/images/test/20160331_NTU_00017.png",
    freeze_level=5,
    pretrained_weights="yolov8n.pt",
    epochs=30,
    imgsz=640,
    batch=64,
    lr=0.001,
    optimizer="Adam"
)

# %%
# 📊 Evaluate metrics for a single trained model
import numpy as np

model_name = "frz5_adam_final"
metrics = evaluate_detector_metrics(f"runs/detect/{model_name}/weights/best.pt")

# 📋 Print evaluation results
print(f"\n🔍 Evaluation for model: {model_name}")
print("📊 Metrics:")
for k, v in metrics.items():
    print(f"   {k}: {v}")

# %% [markdown]
# ### 🥇 3. Best Performing Model
# 
# After analyzing initial experiments, the model trained with `Freeze-5`, `Adam`, `lr=0.001`, `imgsz=640`, and `batch size = 16` had emerged as the top performer with 41.0% accuracy and 5.65 MSE.
# 
# However, based on insights from our configuration-level comparisons, it was hypothesized that increasing the batch size to **64** could further improve model stability and accuracy under the same hyperparameter setting.
# 
# 🔁 **New experiment conducted:**
# - **Freeze Strategy:** `Freeze-5`  
# - **Optimizer:** `Adam`  
# - **Learning Rate:** `lr001`  
# - **Image Size:** `imgsz640`  
# - **Batch Size:** `64`  
# - **Model Name:** `frz5_adam_final`
# 
# 📊 **Evaluation Results:**
# 
# | Metric       | Value     |
# |--------------|-----------|
# | Accuracy     | **44.5%** |
# | MSE          | **5.48**  |
# | Precision    | 0.989     |
# | Recall       | 0.991     |
# 
# ---
# 
# #### ✅ Why This Updated Configuration Performed Best
# 
# - 🔧 **Adam Optimizer:** Maintains adaptive learning across both frozen and trainable layers.
# - ❄️ **Freeze-5:** Retains base features while allowing adaptation to counting task.
# - 🧠 **Larger Batch Size (64):** Provided smoother gradients and improved generalization.
# - 🔍 **High-Resolution Input (640):** Enabled better spatial resolution and object localization.
# 
# > 🏁 This model delivered the highest accuracy and lowest MSE of all experiments, validating the hypothesis drawn from earlier observations. It should be considered the **final recommended configuration** for car counting using YOLOv8 under resource-balanced constraints.
# 
# ---
# 
# ### 🧠 4. Global Summary
# 
# After evaluating all configurations and training a refined model based on analysis, the following insights were confirmed:
# 
# | ✅ What Worked Well                  | ❌ What to Avoid                      |
# |------------------------------------|--------------------------------------|
# | **Freeze-5** (low cost, high perf) | **Freeze-21** (extreme underfitting) |
# | **Adam** optimizer                 | **SGD** optimizer                    |
# | **Image size = 640**              | Image size = 416                     |
# | **Learning rate = 0.001 (`lr001`)** | Learning rate = 0.0001 (`lr0001`)    |
# | **Batch size = 64** (stable under freezing) | Batch size = 16 (less stable in deep freezing) |
# 
# ---
# 
# ### 📌 Final Recommendation
# 
# To achieve the best performance with reasonable training cost:
# 
# - ✅ Use: `Freeze-5` + `Adam` + `lr001` + `imgsz640` + `batch64`
# - ⚠️ Avoid: Over-freezing (e.g., `Freeze-21`), low-resolution inputs, or extremely low learning rates
# 
# This configuration yields:
# - The **highest accuracy (44.5%)**
# - The **lowest MSE (5.48)**
# - Excellent **precision (0.989)** and **recall (0.991)**
# 
# > 🎯 **Bottom Line:**  
# If you're building an object counting system using YOLOv8, this configuration offers the best trade-off between performance, training stability, and efficiency.

# %% [markdown]
# # **What other types of methods can be used to solve similar object counting problems?**

# %% [markdown]
# While object detection-based models like YOLO are highly effective for counting and localizing objects, several alternative approaches can be employed depending on the scenario:
# 
# ---
# 
# ### 1. 📊 Density Map Estimation
# 
# - A CNN is trained to generate a **density map**, where the **integral over the map equals the object count**.
# - Particularly effective in **crowded scenes** with overlapping objects (e.g., people in a crowd, vehicles in traffic).
# - Offers smooth, spatial distribution of object likelihood without requiring exact bounding boxes.
# 
# **Examples:** CSRNet, MCNN
# 
# > ✅ Best for: **Crowded, dense scenes where localization is difficult**
# 
# ---
# 
# ### 2. 🔢 Regression-Based Counting
# 
# - A CNN directly **regresses the total object count** from the input image.
# - Fast and computationally lightweight, but lacks spatial information or localization.
# - Useful in applications where **count is enough**, and detection is unnecessary.
# 
# > ✅ Best for: **Simple counting tasks without needing object positions**
# 
# ---
# 
# ### 3. 🎯 Segmentation + Component Analysis
# 
# - First, a **segmentation model** (e.g., U-Net, Mask R-CNN) is used to extract object regions.
# - Then, **connected components** or blobs are counted using morphological operations.
# - More computationally involved but allows visual inspection of object regions.
# 
# > ✅ Best for: **Precise shape-aware counting (e.g., cells, grains, leaves)**
# 
# ---
# 
# ### 4. 🧱 Classical Computer Vision Techniques
# 
# - Traditional CV methods like **blob detection**, **contour finding**, or **thresholding** (e.g., using OpenCV).
# - Requires good lighting and clean backgrounds.
# - Low-cost and interpretable, but **fragile under real-world variation**.
# 
# > ✅ Best for: **Simple, controlled environments**
# 
# ---
# 
# ### ⚖️ Method Comparison Summary
# 
# | Method                     | Pros                                  | Cons                                 | Best Use Case                          |
# |---------------------------|----------------------------------------|--------------------------------------|----------------------------------------|
# | Density Estimation        | Handles overlaps, crowd-friendly       | No object-level localization         | Crowds, dense scenes                   |
# | Regression                | Fast, low complexity                   | No spatial feedback                  | Basic counting (e.g., trees, vehicles) |
# | Segmentation + Components | Visual output + accurate regions       | Requires post-processing             | Shape-aware domains                    |
# | Classical CV              | Simple, fast, interpretable            | Not robust to real-world variance    | Clean, simple datasets                 |
# | Object Detection (YOLO)   | Localization + count in one model      | Struggles in high occlusion scenes   | General-purpose, moderate density      |
# 
# ---
# 
# > 📌 **Conclusion:**  
# Choose your method based on **density**, **occlusion**, **need for localization**, and **compute constraints**. Detection-based approaches like YOLO are versatile, but alternatives may excel in specialized settings.


