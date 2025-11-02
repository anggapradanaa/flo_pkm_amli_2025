"""
Script untuk capture frame dari CCTV untuk ground truth dataset
Simplified version: 50 frame per lokasi, 2 lokasi
"""

import cv2
import time
import os
from datetime import datetime

# ==============================================================================
# KONFIGURASI
# ==============================================================================

# Pilih 2 CCTV untuk testing (index dari app.py)
CCTV_CONFIGS = [
    {
        'name': 'Polda',
        'url': 'https://livepantau.semarangkota.go.id/5f4862ff-ea80-45d4-8baf-fbd025784710/index.m3u8',
        'code': 'polda'
    }
]

OUTPUT_DIR = "data/ground_truth_frames"
FRAMES_PER_LOCATION = 30  # 30 frame untuk 1 lokasi
CAPTURE_INTERVAL = 20  # Capture setiap 20 detik

# ==============================================================================
# FUNGSI UTAMA
# ==============================================================================

def capture_frames_from_cctv(cctv_config):
    """
    Capture frames dari satu CCTV
    """
    name = cctv_config['name']
    url = cctv_config['url']
    code = cctv_config['code']
    
    print(f"\n{'='*70}")
    print(f"📹 Memulai capture dari: {name}")
    print(f"{'='*70}")
    
    # Buat folder output jika belum ada
    location_dir = os.path.join(OUTPUT_DIR, code)
    os.makedirs(location_dir, exist_ok=True)
    
    # Inisialisasi video capture
    print(f"🔗 Connecting to stream: {url}")
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Tidak bisa connect ke stream {name}")
        return False
    
    print(f"✅ Connected successfully!")
    
    frame_count = 0
    capture_times = []
    
    while frame_count < FRAMES_PER_LOCATION:
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠️  Frame read failed, retrying...")
            time.sleep(2)
            continue
        
        # Generate filename dengan timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{code}_{timestamp}_{frame_count:03d}.jpg"
        filepath = os.path.join(location_dir, filename)
        
        # Save frame
        cv2.imwrite(filepath, frame)
        
        # Tentukan waktu (pagi/siang/sore berdasarkan jam)
        hour = datetime.now().hour
        if 6 <= hour < 11:
            time_period = "pagi"
        elif 11 <= hour < 15:
            time_period = "siang"
        else:
            time_period = "sore"
        
        capture_times.append({
            'filename': filename,
            'timestamp': timestamp,
            'time_period': time_period,
            'location': name
        })
        
        frame_count += 1
        
        # Progress indicator
        progress = (frame_count / FRAMES_PER_LOCATION) * 100
        print(f"📸 Captured: {filename} [{frame_count}/{FRAMES_PER_LOCATION}] ({progress:.1f}%) - {time_period}")
        
        # Wait before next capture
        time.sleep(CAPTURE_INTERVAL)
    
    cap.release()
    
    print(f"\n✅ Capture selesai untuk {name}!")
    print(f"📁 Total frames: {frame_count}")
    print(f"📂 Saved to: {location_dir}")
    
    return capture_times


def generate_csv_template(all_capture_data):
    """
    Generate CSV template untuk manual annotation
    """
    csv_path = "data/ground_truth_counts.csv"
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("filename,lokasi,waktu,mobil,motor,bus,truk,total\n")
        
        # Data rows (empty, untuk diisi manual)
        for data in all_capture_data:
            f.write(f"{data['filename']},{data['location']},{data['time_period']},,,,,\n")
    
    print(f"\n📄 CSV template generated: {csv_path}")
    print(f"⚠️  PENTING: Silakan isi kolom mobil, motor, bus, truk, total secara MANUAL!")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 SCRIPT CAPTURE GROUND TRUTH FRAMES")
    print("="*70)
    print(f"📍 Lokasi: {len(CCTV_CONFIGS)} CCTV")
    print(f"📸 Target: {FRAMES_PER_LOCATION} frame per lokasi")
    print(f"⏱️  Interval: {CAPTURE_INTERVAL} detik")
    print(f"⏳ Estimasi waktu: ~{(FRAMES_PER_LOCATION * CAPTURE_INTERVAL * len(CCTV_CONFIGS)) / 60:.1f} menit")
    print("="*70)
    
    input("\nTekan ENTER untuk mulai capture...")
    
    all_capture_data = []
    
    # Capture dari semua CCTV
    for cctv_config in CCTV_CONFIGS:
        capture_data = capture_frames_from_cctv(cctv_config)
        if capture_data:
            all_capture_data.extend(capture_data)
        
        print("\n⏸️  Waiting 10 seconds before next location...")
        time.sleep(10)
    
    # Generate CSV template
    if all_capture_data:
        generate_csv_template(all_capture_data)
    
    print("\n" + "="*70)
    print("✅ SEMUA CAPTURE SELESAI!")
    print("="*70)
    print(f"📊 Total frames captured: {len(all_capture_data)}")
    print(f"📂 Location: data/ground_truth_frames/")
    print(f"📄 CSV template: data/ground_truth_counts.csv")
    print("\n🔴 LANGKAH SELANJUTNYA:")
    print("   1. Buka setiap frame di folder ground_truth_frames/")
    print("   2. Hitung manual jumlah kendaraan (mobil, motor, bus, truk)")
    print("   3. Isi data ke file ground_truth_counts.csv")
    print("   4. Jalankan test_detection_accuracy.py")
    print("="*70)