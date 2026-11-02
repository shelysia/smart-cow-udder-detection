import time
import os
import requests
import RPi.GPIO as GPIO
import board
import busio
import cv2
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
from smbus2 import SMBus
from ultralytics import YOLO

ESP32_IP = "10.34.209.168"
ESP32_URL = f"http://{ESP32_IP}/capture"

TOKEN_BOT = '8523774456:AAGUMcoFTsddgri6acUfhSNtb_fp2PQzMq8'
CHAT_ID   = '1168668627'

MODEL_PATH = "/home/pi/proyek_ta/best.pt" 
IMG_PATH   = "/home/pi/proyek_ta/capture.jpg"
IMG_RESULT = "/home/pi/proyek_ta/hasil_deteksi.jpg"

BTN_PIN = 17
MLX_ADDR = 0x5A
OFFSET_SUHU = 0.55

CONFIDENCE_THRESHOLD = 0.50
IMAGE_SIZE = 320 

TEMP_MIN_JARAK  = 34.0
TEMP_NORM_MIN   = 34.5
TEMP_NORM_MAX   = 37.5
TEMP_MAST_MIN   = 38.0
TEMP_MAST_MAX   = 42.0

GPIO.setmode(GPIO.BCM)
GPIO.setup(BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)
oled.fill(0); oled.show()
width = oled.width
height = oled.height
image = Image.new("1", (width, height))
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

bus = SMBus(1)

print("Loading Model AI...")
try:
    model = YOLO(MODEL_PATH, task='detect')
    print("Model Loaded Success.")
except Exception as e:
    print(f"Error Load Model: {e}")
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((0, 0), "MODEL ERROR", font=font, fill=255)
    oled.image(image); oled.show()
    exit()

def update_oled(baris1, baris2="", baris3=""):
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((0, 0),  str(baris1), font=font, fill=255)
    draw.text((0, 20), str(baris2), font=font, fill=255)
    draw.text((0, 40), str(baris3), font=font, fill=255)
    oled.image(image)
    oled.show()

def baca_suhu():
    try:
        data = bus.read_word_data(MLX_ADDR, 0x07)
        raw_temp = (data * 0.02) - 273.15
        final_temp = raw_temp + OFFSET_SUHU
        return final_temp
    except:
        return 0.0

def ambil_foto_esp32():
    try:
        response = requests.get(ESP32_URL, timeout=8)
        if response.status_code == 200:
            with open(IMG_PATH, "wb") as f:
                f.write(response.content)
            return True
        else:
            return False
    except:
        return False

def analisa_gambar_yolo():
    try:
        if not os.path.exists(IMG_PATH) or os.path.getsize(IMG_PATH) == 0:
            return "ERROR"

        results = model.predict(source=IMG_PATH, conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE, save=False, verbose=False)
        
        if len(results[0].boxes) == 0:
            return "NIHIL" 
        
        res_plotted = results[0].plot()
        cv2.imwrite(IMG_RESULT, res_plotted)
        
        top_box = results[0].boxes[0]
        class_id = int(top_box.cls[0])
        
        if class_id == 1:
            return "NORMAL"
        elif class_id == 0:
            return "ABNORMAL"
        else:
            return "UNKNOWN"
    except Exception as e:
        print(f"Error AI: {e}")
        return "ERROR"

def kirim_telegram(judul, status, suhu, note, file_gambar):
    caption = (
        f"🔔 <b>{judul}</b>\n"
        f"------------------------\n"
        f"🩺 <b>Visual AI:</b> {status}\n"
        f"🌡 <b>Suhu:</b> {suhu:.2f}°C\n"
        f"📝 <b>Note:</b> {note}"
    )
    url = f"https://api.telegram.org/bot{TOKEN_BOT}/sendPhoto"
    try:
        if not os.path.exists(file_gambar): return
        files = {'photo': open(file_gambar, 'rb')}
        data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'}
        requests.post(url, files=files, data=data)
    except:
        pass

print("SISTEM SIAP DIGUNAKAN")
update_oled("SISTEM SIAP", "Tekan: Scan", "Tahan: OFF")

try:
    while True:
        if GPIO.input(BTN_PIN) == False:
            print("Tombol ditekan, menghitung durasi...")
            start_time = time.time()
            shutdown_triggered = False
            
            while GPIO.input(BTN_PIN) == False:
                durasi = time.time() - start_time
                
                if durasi > 3:
                    print("LONG PRESS DETECTED: SHUTDOWN")
                    update_oled("MEMATIKAN...", "Sampai Jumpa", "Safe Shutdown")
                    time.sleep(2)
                    os.system("sudo shutdown now") 
                    shutdown_triggered = True
                    break 
                
                time.sleep(0.1)
            
            if not shutdown_triggered:
                print("KLIK SINGKAT: MULAI SCAN")
                
                update_oled("MENGHUBUNGKAN...", "Kamera ESP32", "")
                
                if ambil_foto_esp32():
                    update_oled("ANALISA AI...", "Memproses...", "")
                    hasil_img = analisa_gambar_yolo()
                    temp = baca_suhu()
                    
                    diagnosis_final = "INDIKASI"
                    file_kirim = IMG_RESULT
                    judul_laporan = "HASIL SCAN"
                    pesan_note = f"Suhu: {temp:.1f}C"

                    if hasil_img == "ERROR":
                        diagnosis_final = "SISTEM ERROR"
                        judul_laporan = "AI GAGAL"
                        file_kirim = IMG_PATH
                    elif hasil_img == "NIHIL":
                        diagnosis_final = "NO OBJECT"
                        judul_laporan = "BUKAN AMBING"
                        file_kirim = IMG_PATH
                    elif temp < TEMP_MIN_JARAK:
                        diagnosis_final = "DEKATKAN ALAT"
                        judul_laporan = "JARAK JAUH"
                    elif (hasil_img == "NORMAL") and (temp <= TEMP_NORM_MAX):
                        diagnosis_final = "SEHAT"
                        judul_laporan = "SEHAT"
                    elif (hasil_img == "ABNORMAL") and (temp >= TEMP_MAST_MIN):
                        diagnosis_final = "MASTITIS (+)"
                        judul_laporan = "MASTITIS"
                    
                    print(f"Hasil: {diagnosis_final}")
                    disp_ai = hasil_img if len(hasil_img) < 8 else hasil_img[:3]
                    update_oled(f"{diagnosis_final}", f"Suhu: {temp:.1f} C", f"AI: {disp_ai}")
                    
                    kirim_telegram(judul_laporan, diagnosis_final, temp, pesan_note, file_kirim)
                    
                    time.sleep(4)
                    update_oled("SISTEM SIAP", "Tekan: Scan", "Tahan: OFF")
                else:
                    update_oled("GAGAL KONEKSI", "Cek ESP32", "")
                    time.sleep(2)
                    update_oled("SISTEM SIAP", "Tekan: Scan", "Tahan: OFF")
            
            time.sleep(0.5)

        time.sleep(0.1)

except KeyboardInterrupt:
    GPIO.cleanup()
    oled.fill(0); oled.show()
    print("Program Berhenti.")