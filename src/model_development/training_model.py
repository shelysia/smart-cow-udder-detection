
# -*- coding: utf-8 -*-
"""TRAINING MODEL DETEKSI MASTITIS SAPI

Metode: YOLOv8n | Dataset: Roboflow | Lingkungan: Google Colab

Alur:
1. Instalasi dan setup lingkungan
2. Download dataset dari Roboflow
3. Validasi konfigurasi data.yaml
4. Visualisasi statistik dataset
5. Visualisasi sampel gambar dan bounding box
6. Training YOLOv8n (75 Epoch)
7. Evaluasi Validation dan Test
8. Export best.pt ke TensorFlow Lite INT8
"""

# ============================================================
# BAGIAN 1: INSTALASI & SETUP LINGKUNGAN
# ============================================================

print("Menginstall library yang dibutuhkan...")

!pip install roboflow ultralytics seaborn -q

import os
import glob
import yaml
import cv2
import random
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO
from IPython.display import Image, display
from google.colab import userdata, drive, files

# ============================================================
# BAGIAN 2: KONFIGURASI PROYEK
# ============================================================

WORKSPACE_NAME = "cow-udder"
PROJECT_NAME = "deteksi-mastitis-sapi"
DATASET_VERSION = 3

MODEL_BASE = "yolov8n.pt"

EPOCHS = 75
PATIENCE = 10
BATCH_SIZE = 16
IMAGE_SIZE = 320

# Lokasi penyimpanan Google Drive
drive.mount('/content/drive')

path_drive = '/content/drive/MyDrive/Skripsi_Sapi'

# Nama folder baru agar hasil sebelumnya tidak tertimpa
run_name = 'hasil_training_75epoch_v2'

run_dir = os.path.join(path_drive, run_name)

os.makedirs(path_drive, exist_ok=True)

# ============================================================
# BAGIAN 3: DOWNLOAD DATASET DARI ROBOFLOW
# ============================================================

print("\nMenyiapkan Dataset dari Roboflow...")

api_key = userdata.get("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError(
        "ROBOFLOW_API_KEY tidak ditemukan di Colab Secrets."
    )

rf = Roboflow(api_key=api_key)

project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)

version = project.version(DATASET_VERSION)

dataset = version.download("yolov8")

base_path = dataset.location

print(f"Dataset berhasil diunduh: {base_path}")

# ============================================================
# BAGIAN 4: VALIDASI DAN PERBAIKAN data.yaml
# ============================================================

print("\nMemeriksa konfigurasi dataset...")

yaml_path = os.path.join(base_path, "data.yaml")

with open(yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Menentukan folder training
if os.path.exists(f"{base_path}/train/images"):
    data_config['train'] = "train/images"
else:
    data_config['train'] = "train"

# Menentukan folder validation
if os.path.exists(f"{base_path}/valid/images"):
    data_config['val'] = "valid/images"
elif os.path.exists(f"{base_path}/val/images"):
    data_config['val'] = "val/images"
else:
    raise FileNotFoundError(
        "Folder validation tidak ditemukan. "
        "Training dihentikan agar tidak menggunakan data test/train sebagai validation."
    )

# Menentukan folder test
if os.path.exists(f"{base_path}/test/images"):
    data_config['test'] = "test/images"

data_config['path'] = base_path

with open(yaml_path, 'w') as f:
    yaml.safe_dump(data_config, f, sort_keys=False)

print("Konfigurasi dataset berhasil diperbarui.")
print("Train:", data_config['train'])
print("Validation:", data_config['val'])
print("Test:", data_config.get('test', 'Tidak tersedia'))
print("Nama kelas:", data_config['names'])

# ============================================================
# BAGIAN 5: VISUALISASI DISTRIBUSI DATASET
# ============================================================

print("\nMembuat Visualisasi Statistik Dataset...")

splits = ['train', 'valid', 'test']

split_counts = {}
class_names = data_config['names']

if isinstance(class_names, list):
    labels_map = {
        i: name for i, name in enumerate(class_names)
    }
else:
    labels_map = {
        int(k): v for k, v in class_names.items()
    }

class_counts = {
    class_id: 0 for class_id in labels_map
}

for split in splits:

    image_folder = os.path.join(base_path, split, 'images')
    label_folder = os.path.join(base_path, split, 'labels')

    if not os.path.exists(image_folder):
        continue

    imgs = (
        glob.glob(f"{image_folder}/*.jpg") +
        glob.glob(f"{image_folder}/*.jpeg") +
        glob.glob(f"{image_folder}/*.png")
    )

    split_counts[split] = len(imgs)

    txts = glob.glob(f"{label_folder}/*.txt")

    for txt in txts:
        with open(txt, 'r') as f:
            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                try:
                    class_id = int(parts[0])

                    if class_id in class_counts:
                        class_counts[class_id] += 1

                except (ValueError, IndexError):
                    continue

if not split_counts:
    raise RuntimeError("Tidak ditemukan gambar dalam dataset.")

# Visualisasi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.pie(
    split_counts.values(),
    labels=split_counts.keys(),
    autopct='%1.1f%%',
    startangle=140
)

plt.title('Pembagian Dataset', fontweight='bold')

plt.subplot(1, 2, 2)

classes_viz = [
    labels_map[k] for k in class_counts.keys()
]

values_viz = list(class_counts.values())

sns.barplot(
    x=classes_viz,
    y=values_viz
)

plt.title('Distribusi Kelas', fontweight='bold')
plt.ylabel('Jumlah Anotasi')

for i, v in enumerate(values_viz):
    plt.text(i, v + 2, str(v), ha='center')

plt.tight_layout()
plt.show()

print("Visualisasi Statistik Selesai.")

# ============================================================
# BAGIAN 6: VISUALISASI SAMPEL GAMBAR
# ============================================================

print("\nMenampilkan 5 Sampel Acak dengan Bounding Box...")

image_folder = os.path.join(base_path, "train/images")
label_folder = os.path.join(base_path, "train/labels")

image_files = (
    glob.glob(f"{image_folder}/*.jpg") +
    glob.glob(f"{image_folder}/*.jpeg") +
    glob.glob(f"{image_folder}/*.png")
)

random_samples = random.sample(
    image_files,
    min(5, len(image_files))
)

if random_samples:

    fig, axes = plt.subplots(
        1, len(random_samples),
        figsize=(4 * len(random_samples), 5)
    )

    if len(random_samples) == 1:
        axes = [axes]

    for i, (img_file, ax) in enumerate(
        zip(random_samples, axes)
    ):

        img = cv2.imread(img_file)

        if img is None:
            ax.set_title("Gambar tidak dapat dibaca")
            ax.axis('off')
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape

        ax.imshow(img)

        label_file = os.path.join(
            label_folder,
            os.path.splitext(os.path.basename(img_file))[0] + ".txt"
        )

        if os.path.exists(label_file):

            with open(label_file, 'r') as f:

                for line in f:
                    parts = line.strip().split()

                    if len(parts) < 5:
                        continue

                    try:
                        cls_id = int(parts[0])

                        x_c, y_c, bw, bh = map(
                            float, parts[1:5]
                        )

                    except ValueError:
                        continue

                    x1 = (x_c - bw / 2) * w
                    y1 = (y_c - bh / 2) * h

                    rect = patches.Rectangle(
                        (x1, y1),
                        bw * w,
                        bh * h,
                        linewidth=2,
                        edgecolor='yellow',
                        facecolor='none'
                    )

                    ax.add_patch(rect)

                    ax.text(
                        x1,
                        max(y1 - 5, 0),
                        labels_map.get(cls_id, str(cls_id)),
                        color='white',
                        fontsize=9,
                        bbox=dict(
                            facecolor='red',
                            alpha=0.5
                        )
                    )

        ax.axis('off')
        ax.set_title(f"Sampel {i+1}")

    plt.tight_layout()
    plt.show()

else:
    print("Tidak ada gambar training.")

# ============================================================
# BAGIAN 7: TRAINING YOLOv8n
# ============================================================

print("\n======================================")
print("MEMULAI TRAINING YOLOv8n")
print("======================================")

model = YOLO(MODEL_BASE)

results = model.train(
    data=yaml_path,
    epochs=EPOCHS,
    patience=PATIENCE,
    batch=BATCH_SIZE,
    imgsz=IMAGE_SIZE,
    plots=True,
    save=True,
    project=path_drive,
    name=run_name,
    exist_ok=True
)

best_model_path = os.path.join(
    run_dir,
    'weights',
    'best.pt'
)

last_model_path = os.path.join(
    run_dir,
    'weights',
    'last.pt'
)

if not os.path.exists(best_model_path):
    raise FileNotFoundError(
        f"best.pt tidak ditemukan: {best_model_path}"
    )

print("\nTraining Selesai!")
print("Best model:", best_model_path)
print("Last model tersedia:", os.path.exists(last_model_path))

# ============================================================
# BAGIAN 8: EVALUASI MODEL
# ============================================================

print("\nMengevaluasi model...")

best_model = YOLO(best_model_path)

# Evaluasi validation
val_results = best_model.val(
    data=yaml_path,
    split='val',
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    plots=True,
    project=path_drive,
    name='evaluasi_validation_75epoch',
    exist_ok=True
)

print("\nHASIL VALIDATION")
print(f"Precision : {val_results.box.mp:.4f}")
print(f"Recall    : {val_results.box.mr:.4f}")
print(f"mAP50     : {val_results.box.map50:.4f}")
print(f"mAP50-95  : {val_results.box.map:.4f}")

# Evaluasi test jika tersedia
if data_config.get('test'):

    test_results = best_model.val(
        data=yaml_path,
        split='test',
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        plots=True,
        project=path_drive,
        name='evaluasi_test_75epoch',
        exist_ok=True
    )

    print("\nHASIL TEST")
    print(f"Precision : {test_results.box.mp:.4f}")
    print(f"Recall    : {test_results.box.mr:.4f}")
    print(f"mAP50     : {test_results.box.map50:.4f}")
    print(f"mAP50-95  : {test_results.box.map:.4f}")

else:
    print("Split test tidak tersedia. Evaluasi test dilewati.")

# ============================================================
# BAGIAN 9: EXPORT KE TENSORFLOW LITE INT8
# ============================================================

print("\nMengubah model terbaik ke TensorFlow Lite INT8...")

exported_path = best_model.export(
    format='tflite',
    imgsz=IMAGE_SIZE,
    int8=True,
    data=yaml_path
)

exported_path = Path(exported_path)

if not exported_path.exists():
    raise FileNotFoundError(
        f"File hasil export tidak ditemukan: {exported_path}"
    )

model_output_dir = os.path.join(path_drive, 'model')

os.makedirs(model_output_dir, exist_ok=True)

final_tflite_path = os.path.join(
    model_output_dir,
    'modelta.tflite'
)

shutil.copy2(
    exported_path,
    final_tflite_path
)

print("Export TFLite berhasil!")
print("Lokasi:", final_tflite_path)

# Download hasil TFLite
try:
    files.download(final_tflite_path)
except Exception as e:
    print("Auto-download gagal:", e)
    print("Silakan ambil file dari Google Drive.")

# ============================================================
# BAGIAN 10: TAMPILKAN GRAFIK TRAINING
# ============================================================

results_png = os.path.join(
    run_dir,
    'results.png'
)

if os.path.exists(results_png):
    display(Image(filename=results_png))
else:
    print("Grafik results.png tidak ditemukan.")

print("\n======================================")
print("SELURUH PROSES SELESAI")
print("======================================")
print("Epoch:", EPOCHS)
print("Best model:", best_model_path)
print("TFLite:", final_tflite_path)
