
import os
import time
import requests
import numpy as np
from PIL import Image, ImageDraw

# Import TFLite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


# ==================================================
# 1. KONFIGURASI
# ==================================================

BASE_DIR = "/home/pi/proyek_ta"

MODEL_PATH = os.path.join(BASE_DIR, "modelta.tflite")
DATA_DIR = os.path.join(BASE_DIR, "sapi")

SUB_FOLDERS = ["sapi a", "sapi b"]

# Sesuai urutan kelas dari model training
CLASS_NAMES = {
    0: "ABNORMAL",
    1: "NORMAL"
}

# Parameter deteksi
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Telegram: atur melalui environment variable
TOKEN_BOT = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ==================================================
# 2. MEMUAT MODEL TFLITE
# ==================================================

print(f"Memuat model: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

interpreter = tflite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_info = input_details[0]
output_info = output_details[0]

input_shape = input_info["shape"]

input_height = int(input_shape[1])
input_width = int(input_shape[2])
input_dtype = input_info["dtype"]

print("\nInformasi Model:")
print("Input shape :", input_shape)
print("Input dtype :", input_dtype)
print("Output shape:", output_info["shape"])
print("Output dtype:", output_info["dtype"])

print(
    f"Ukuran input: {input_width} x {input_height}"
)


# ==================================================
# 3. FUNGSI KUANTISASI TFLITE
# ==================================================

def dequantize_tensor(data, details):
    """
    Mengubah output INT8/UINT8 menjadi float
    berdasarkan parameter kuantisasi TFLite.
    """

    if data.dtype not in (np.int8, np.uint8):
        return data.astype(np.float32)

    scale, zero_point = details["quantization"]

    if scale == 0:
        return data.astype(np.float32)

    return (
        data.astype(np.float32) - zero_point
    ) * scale


def prepare_input(image):
    """
    Resize gambar menggunakan letterbox
    agar rasio gambar tetap terjaga.
    """

    original_width, original_height = image.size

    scale = min(
        input_width / original_width,
        input_height / original_height
    )

    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (input_width, input_height),
        (114, 114, 114)
    )

    pad_x = (input_width - new_width) // 2
    pad_y = (input_height - new_height) // 2

    canvas.paste(resized, (pad_x, pad_y))

    # YOLO menggunakan input RGB dengan rentang 0-1
    image_array = np.asarray(
        canvas,
        dtype=np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Sesuaikan dengan tipe input TFLite
    if input_dtype in (np.int8, np.uint8):

        scale_q, zero_point_q = input_info["quantization"]

        if scale_q <= 0:
            raise ValueError(
                "Parameter kuantisasi input tidak valid."
            )

        image_array = np.round(
            image_array / scale_q + zero_point_q
        )

        limits = np.iinfo(input_dtype)

        image_array = np.clip(
            image_array,
            limits.min,
            limits.max
        ).astype(input_dtype)

    else:
        image_array = image_array.astype(input_dtype)

    return (
        image_array,
        scale,
        pad_x,
        pad_y,
        original_width,
        original_height
    )


# ==================================================
# 4. FUNGSI NON-MAXIMUM SUPPRESSION
# ==================================================

def calculate_iou(box, boxes):
    """
    Menghitung Intersection over Union.
    Format box: x1, y1, x2, y2.
    """

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    area_box = (
        (box[2] - box[0]) *
        (box[3] - box[1])
    )

    area_boxes = (
        (boxes[:, 2] - boxes[:, 0]) *
        (boxes[:, 3] - boxes[:, 1])
    )

    union = area_box + area_boxes - intersection

    return intersection / np.maximum(union, 1e-7)


def non_max_suppression(boxes, scores, iou_threshold):
    """
    Menghilangkan bounding box yang terlalu tumpang tindih.
    """

    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    selected = []

    while len(order) > 0:

        current = order[0]
        selected.append(current)

        if len(order) == 1:
            break

        remaining = order[1:]

        ious = calculate_iou(
            boxes[current],
            boxes[remaining]
        )

        order = remaining[
            ious < iou_threshold
        ]

    return selected


# ==================================================
# 5. MEMPROSES OUTPUT YOLOV8
# ==================================================

def process_yolo_output(output_data, scale, pad_x, pad_y,
                         original_width, original_height):
    """
    Memproses output YOLOv8 standar:
    4 koordinat bounding box + 2 skor kelas.

    Mendukung bentuk:
    [1, 6, jumlah_prediksi]
    atau
    [1, jumlah_prediksi, 6]
    """

    predictions = np.squeeze(output_data)

    if predictions.ndim != 2:
        raise ValueError(
            f"Output YOLO tidak sesuai: {predictions.shape}"
        )

    # Pastikan format menjadi [jumlah_prediksi, 6]
    if predictions.shape[0] == 6:
        predictions = predictions.T

    elif predictions.shape[1] != 6:
        raise ValueError(
            "Output model tidak memiliki 6 nilai per prediksi. "
            f"Bentuk output: {predictions.shape}"
        )

    # 4 nilai pertama: x_center, y_center, width, height
    # 2 nilai berikutnya: skor abnormal dan normal
    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:6]

    # Cari kelas dengan skor tertinggi
    class_ids = np.argmax(class_scores, axis=1)
    confidence_scores = np.max(class_scores, axis=1)

    # Filter berdasarkan confidence
    valid = confidence_scores >= CONF_THRESHOLD

    boxes_xywh = boxes_xywh[valid]
    class_ids = class_ids[valid]
    confidence_scores = confidence_scores[valid]

    if len(boxes_xywh) == 0:
        return []

    # Konversi xywh menjadi xyxy
    boxes = np.zeros_like(boxes_xywh)

    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # Beberapa ekspor dapat menghasilkan koordinat ternormalisasi.
    # Jika demikian, ubah ke satuan piksel input.
    if np.max(np.abs(boxes)) <= 2.0:
        boxes[:, [0, 2]] *= input_width
        boxes[:, [1, 3]] *= input_height

    # Kembalikan koordinat ke ukuran gambar asli
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y

    boxes /= scale

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]], 0, original_width
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]], 0, original_height
    )

    results = []

    # NMS dilakukan per kelas
    for class_id in np.unique(class_ids):

        class_mask = class_ids == class_id

        class_boxes = boxes[class_mask]
        class_confidences = confidence_scores[class_mask]

        indices = non_max_suppression(
            class_boxes,
            class_confidences,
            IOU_THRESHOLD
        )

        for index in indices:

            results.append({
                "class_id": int(class_id),
                "label": CLASS_NAMES[int(class_id)],
                "confidence": float(class_confidences[index]),
                "box": class_boxes[index].tolist()
            })

    return results


# ==================================================
# 6. FUNGSI ANALISIS GAMBAR
# ==================================================

def analisa_gambar(file_path):

    try:
        with Image.open(file_path) as source:
            image = source.convert("RGB")

        (
            input_data,
            scale,
            pad_x,
            pad_y,
            original_width,
            original_height
        ) = prepare_input(image)

        interpreter.set_tensor(
            input_info["index"],
            input_data
        )

        interpreter.invoke()

        raw_output = interpreter.get_tensor(
            output_info["index"]
        )

        output_data = dequantize_tensor(
            raw_output,
            output_info
        )

        detections = process_yolo_output(
            output_data,
            scale,
            pad_x,
            pad_y,
            original_width,
            original_height
        )

        return image, detections

    except Exception as e:
        print(f"Error analisis {file_path}: {e}")
        return None, None


# ==================================================
# 7. FUNGSI VISUALISASI HASIL DETEKSI
# ==================================================

def simpan_hasil_gambar(image, detections, output_path):

    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        label = detection["label"]
        confidence = detection["confidence"] * 100

        # Warna kotak berdasarkan kelas
        color = (
            "red" if label == "ABNORMAL"
            else "green"
        )

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=4
        )

        text_label = f"{label} {confidence:.1f}%"

        text_box = draw.textbbox(
            (x1, max(0, y1 - 20)),
            text_label
        )

        draw.rectangle(
            text_box,
            fill=color
        )

        draw.text(
            (x1, max(0, y1 - 20)),
            text_label,
            fill="white"
        )

    result_image.save(output_path)

    return output_path


# ==================================================
# 8. FUNGSI KIRIM TELEGRAM
# ==================================================

def kirim_telegram(nama_sapi, nama_file, detections, image_path):

    if not TOKEN_BOT or not CHAT_ID:
        print("Telegram belum dikonfigurasi. Lewati pengiriman.")
        return

    if any(
        detection["label"] == "ABNORMAL"
        for detection in detections
    ):
        status = "⚠️ TERDETEKSI ABNORMAL"
    elif detections:
        status = "✅ TERDETEKSI NORMAL"
    else:
        status = "ℹ️ TIDAK ADA OBJEK TERDETEKSI"

    caption = (
        f"🧪 <b>HASIL PENGUJIAN YOLOv8n</b>\n"
        f"------------------------\n"
        f"🐄 <b>Sumber:</b> {nama_sapi.upper()}\n"
        f"📂 <b>File:</b> {nama_file}\n"
        f"📝 <b>Status:</b> {status}\n"
    )

    if detections:

        caption += "\n<b>Detail Deteksi:</b>\n"

        for i, detection in enumerate(detections, 1):

            caption += (
                f"{i}. {detection['label']} - "
                f"{detection['confidence'] * 100:.2f}%\n"
            )

    url = (
        f"https://api.telegram.org/"
        f"bot{TOKEN_BOT}/sendPhoto"
    )

    try:
        with open(image_path, "rb") as photo:

            response = requests.post(
                url,
                files={"photo": photo},
                data={
                    "chat_id": CHAT_ID,
                    "caption": caption,
                    "parse_mode": "HTML"
                },
                timeout=30
            )

        response.raise_for_status()

        result = response.json()

        if result.get("ok"):
            print(f"Telegram terkirim: {nama_file}")
        else:
            print(f"Telegram gagal: {result}")

    except Exception as e:
        print(f"Gagal mengirim Telegram: {e}")


# ==================================================
# 9. PROGRAM UTAMA
# ==================================================

print("\n======================================")
print("      MULAI PENGUJIAN YOLOv8n")
print("======================================")

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"Folder data tidak ditemukan: {DATA_DIR}"
    )

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "hasil_pengujian"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

total_files = 0
total_detections = 0
failed_files = 0

for subfolder in SUB_FOLDERS:

    current_path = os.path.join(
        DATA_DIR,
        subfolder
    )

    print(f"\nMembuka folder: {subfolder.upper()}")

    if not os.path.isdir(current_path):
        print("Folder tidak ditemukan, dilewati.")
        continue

    try:
        gambar_list = sorted([
            f for f in os.listdir(current_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

    except Exception as e:
        print(f"Gagal membaca folder: {e}")
        continue

    if not gambar_list:
        print("Tidak ada gambar.")
        continue

    for nama_file in gambar_list:

        full_path = os.path.join(
            current_path,
            nama_file
        )

        print(f"\nMenguji: {nama_file}")

        image, detections = analisa_gambar(full_path)

        if image is None or detections is None:
            failed_files += 1
            continue

        print(f"Jumlah objek terdeteksi: {len(detections)}")

        for detection in detections:

            print(
                f"  - {detection['label']} | "
                f"Confidence: "
                f"{detection['confidence'] * 100:.2f}%"
            )

        output_name = (
            os.path.splitext(nama_file)[0]
            + "_hasil.jpg"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        simpan_hasil_gambar(
            image,
            detections,
            output_path
        )

        kirim_telegram(
            subfolder,
            nama_file,
            detections,
            output_path
        )

        total_files += 1
        total_detections += len(detections)

        time.sleep(3)


# ==================================================
# 10. RINGKASAN
# ==================================================

print("\n======================================")
print("       PENGUJIAN SELESAI")
print("======================================")

print(f"Total gambar berhasil diuji : {total_files}")
print(f"Total objek terdeteksi      : {total_detections}")
print(f"Total gambar gagal diproses : {failed_files}")
print(f"Hasil gambar tersimpan di   : {OUTPUT_DIR}")

print("======================================")
