"""
Script untuk test akurasi deteksi YOLOv8
Input: ground_truth_counts.csv
Output: detection_results.csv + metrics
"""

import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os
from datetime import datetime

# ==============================================================================
# KONFIGURASI
# ==============================================================================

MODEL_PATH = 'yolov8l.pt'
GROUND_TRUTH_CSV = 'data/ground_truth_counts.csv'
OUTPUT_CSV = 'data/detection_results.csv'
GROUND_TRUTH_FRAMES_DIR = 'data/ground_truth_frames'

# Vehicle classes (sama dengan app.py)
ALLOWED_VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

# Confidence threshold (sama dengan app.py)
CONFIDENCE_THRESHOLD = 0.25

# ==============================================================================
# LOAD MODEL
# ==============================================================================

print("="*70)
print("🚀 LOADING YOLO MODEL...")
print("="*70)

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ==============================================================================
# LOAD GROUND TRUTH DATA
# ==============================================================================

print("\n" + "="*70)
print("📄 LOADING GROUND TRUTH DATA...")
print("="*70)

try:
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    print(f"✅ Ground truth loaded: {len(gt_df)} frames")
    print(f"📂 Columns: {list(gt_df.columns)}")
    
    # Validasi kolom yang diperlukan
    required_cols = ['filename', 'lokasi', 'waktu', 'total']
    missing_cols = [col for col in required_cols if col not in gt_df.columns]
    
    if missing_cols:
        print(f"❌ ERROR: Missing columns: {missing_cols}")
        exit(1)
    
    # Filter baris yang sudah diisi (total tidak kosong)
    gt_df = gt_df[gt_df['total'].notna()]
    print(f"✅ Valid annotated frames: {len(gt_df)}")
    
    if len(gt_df) == 0:
        print("❌ ERROR: Tidak ada frame yang sudah di-annotate!")
        print("   Silakan isi ground_truth_counts.csv terlebih dahulu.")
        exit(1)
    
except FileNotFoundError:
    print(f"❌ ERROR: File {GROUND_TRUTH_CSV} tidak ditemukan!")
    print("   Jalankan capture_groundtruth.py terlebih dahulu.")
    exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    exit(1)

# ==============================================================================
# TESTING LOOP
# ==============================================================================

print("\n" + "="*70)
print("🧪 STARTING DETECTION ACCURACY TEST...")
print("="*70)

results = []
processed = 0
skipped = 0

for index, row in gt_df.iterrows():
    filename = row['filename']
    lokasi = row['lokasi']
    waktu = row['waktu']
    gt_total = int(row['total'])
    
    # Determine full path
    # Check both possible locations
    possible_paths = [
        os.path.join(GROUND_TRUTH_FRAMES_DIR, lokasi.lower().replace(' ', ''), filename),
        os.path.join(GROUND_TRUTH_FRAMES_DIR, filename)
    ]
    
    frame_path = None
    for path in possible_paths:
        if os.path.exists(path):
            frame_path = path
            break
    
    if frame_path is None:
        print(f"⚠️  SKIP: File not found - {filename}")
        skipped += 1
        continue
    
    # Read frame
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print(f"⚠️  SKIP: Cannot read - {filename}")
        skipped += 1
        continue
    
    # YOLO Inference
    detections = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    detected_count = 0
    
    # Count vehicles
    for result in detections:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            
            if label in ALLOWED_VEHICLE_CLASSES:
                detected_count += 1
    
    # Calculate errors
    error = detected_count - gt_total
    abs_error = abs(error)
    percentage_error = (abs_error / gt_total * 100) if gt_total > 0 else 0
    
    # Store result
    results.append({
        'filename': filename,
        'lokasi': lokasi,
        'waktu': waktu,
        'ground_truth': gt_total,
        'detected': detected_count,
        'error': error,
        'abs_error': abs_error,
        'percentage_error': percentage_error
    })
    
    processed += 1
    
    # Progress
    status = "✅" if abs_error <= 2 else "⚠️" if abs_error <= 5 else "❌"
    print(f"{status} [{processed}/{len(gt_df)}] {filename}: GT={gt_total}, Detected={detected_count}, Error={error:+d}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

print("\n" + "="*70)
print("💾 SAVING RESULTS...")
print("="*70)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Results saved to: {OUTPUT_CSV}")

# ==============================================================================
# CALCULATE METRICS
# ==============================================================================

print("\n" + "="*70)
print("📊 HASIL PENGUJIAN AKURASI DETEKSI")
print("="*70)

if len(results_df) > 0:
    # Overall metrics
    mae = results_df['abs_error'].mean()
    mse = (results_df['error'] ** 2).mean()
    rmse = np.sqrt(mse)
    total_gt = results_df['ground_truth'].sum()
    total_detected = results_df['detected'].sum()
    accuracy = 100 * (1 - (results_df['abs_error'].sum() / total_gt)) if total_gt > 0 else 0
    
    print(f"\n📈 OVERALL METRICS:")
    print(f"   • Frames tested: {len(results_df)}")
    print(f"   • Frames skipped: {skipped}")
    print(f"   • Mean Absolute Error (MAE): {mae:.2f} kendaraan")
    print(f"   • Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"   • Accuracy: {accuracy:.2f}%")
    print(f"   • Total GT: {total_gt}, Total Detected: {total_detected}")
    
    # Breakdown by location
    print(f"\n📍 BREAKDOWN BY LOKASI:")
    for lokasi in results_df['lokasi'].unique():
        lokasi_df = results_df[results_df['lokasi'] == lokasi]
        lokasi_mae = lokasi_df['abs_error'].mean()
        lokasi_accuracy = 100 * (1 - (lokasi_df['abs_error'].sum() / lokasi_df['ground_truth'].sum()))
        print(f"   • {lokasi}:")
        print(f"     - Frames: {len(lokasi_df)}")
        print(f"     - MAE: {lokasi_mae:.2f}")
        print(f"     - Accuracy: {lokasi_accuracy:.2f}%")
    
    # Breakdown by time
    print(f"\n⏰ BREAKDOWN BY WAKTU:")
    for waktu in results_df['waktu'].unique():
        waktu_df = results_df[results_df['waktu'] == waktu]
        waktu_mae = waktu_df['abs_error'].mean()
        waktu_accuracy = 100 * (1 - (waktu_df['abs_error'].sum() / waktu_df['ground_truth'].sum()))
        print(f"   • {waktu.capitalize()}:")
        print(f"     - Frames: {len(waktu_df)}")
        print(f"     - MAE: {waktu_mae:.2f}")
        print(f"     - Accuracy: {waktu_accuracy:.2f}%")
    
    # Error distribution
    print(f"\n📉 ERROR DISTRIBUTION:")
    perfect = len(results_df[results_df['abs_error'] == 0])
    excellent = len(results_df[results_df['abs_error'] <= 2])
    good = len(results_df[(results_df['abs_error'] > 2) & (results_df['abs_error'] <= 5)])
    poor = len(results_df[results_df['abs_error'] > 5])
    
    print(f"   • Perfect (error = 0): {perfect} frames ({perfect/len(results_df)*100:.1f}%)")
    print(f"   • Excellent (error ≤ 2): {excellent} frames ({excellent/len(results_df)*100:.1f}%)")
    print(f"   • Good (error ≤ 5): {good} frames ({good/len(results_df)*100:.1f}%)")
    print(f"   • Poor (error > 5): {poor} frames ({poor/len(results_df)*100:.1f}%)")
    
    # Save summary
    summary_path = 'data/detection_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("HASIL PENGUJIAN AKURASI DETEKSI YOLOV8\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Frames tested: {len(results_df)}\n")
        f.write(f"MAE: {mae:.2f} kendaraan\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
    
    print(f"\n💾 Summary saved to: {summary_path}")

else:
    print("❌ No results to display!")

print("\n" + "="*70)
print("✅ TEST SELESAI!")
print("="*70)
print(f"📂 Results: {OUTPUT_CSV}")
print(f"📊 Summary: data/detection_summary.txt")
print("\n🔴 LANGKAH SELANJUTNYA:")
print("   Jalankan: python evaluation/visualize_results.py")
print("="*70)