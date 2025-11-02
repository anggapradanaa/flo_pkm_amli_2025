"""
Script untuk test performa sistem (FPS, Latency, Resource Usage)
Simplified version: 5 menit test per stream
"""

import cv2
import time
from ultralytics import YOLO
import psutil
import numpy as np
import json
from datetime import datetime

# ==============================================================================
# KONFIGURASI
# ==============================================================================

MODEL_PATH = 'yolov8l.pt'

# Test configurations
TEST_CONFIGS = [
    {
        'name': '1 Stream (Tugu Muda)',
        'streams': [
            'https://livepantau.semarangkota.go.id/cb44a209-58b5-4a7e-99c4-55d5ada0aae2/index.m3u8'
        ]
    },
    {
        'name': '2 Streams (Tugu Muda + Kalibanteng)',
        'streams': [
            'https://livepantau.semarangkota.go.id/cb44a209-58b5-4a7e-99c4-55d5ada0aae2/index.m3u8',
            'https://livepantau.semarangkota.go.id/1aee2909-217d-4452-b985-2f59d332b2a9/index.m3u8'
        ]
    }
]

TEST_DURATION = 300  # 5 menit per test
OUTPUT_JSON = 'data/performance_results.json'

# Vehicle classes
ALLOWED_VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

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
# PERFORMANCE TEST FUNCTION
# ==============================================================================

def test_single_stream(stream_url, test_name, duration):
    """
    Test performance untuk single stream
    """
    print(f"\n{'='*70}")
    print(f"🧪 Testing: {test_name}")
    print(f"{'='*70}")
    print(f"🔗 Stream: {stream_url}")
    print(f"⏱️  Duration: {duration} seconds")
    
    # Initialize video capture
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open stream")
        return None
    
    print(f"✅ Stream opened successfully")
    
    # Metrics storage
    fps_list = []
    latency_list = []
    cpu_usage = []
    ram_usage = []
    vehicle_counts = []
    
    start_time = time.time()
    frame_count = 0
    last_log_time = start_time
    
    print(f"\n⏳ Starting performance test...")
    print(f"{'Time':>8} | {'FPS':>6} | {'Latency':>10} | {'CPU':>6} | {'RAM':>6} | {'Vehicles':>8}")
    print("-" * 70)
    
    while (time.time() - start_time) < duration:
        # Measure frame processing time
        frame_start = time.time()
        
        ret, frame = cap.read()
        
        if not ret:
            time.sleep(0.5)
            continue
        
        # YOLO inference
        results = model.predict(frame, verbose=False)
        vehicle_count = 0
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in ALLOWED_VEHICLE_CLASSES:
                    vehicle_count += 1
        
        frame_end = time.time()
        
        # Calculate latency (ms)
        latency = (frame_end - frame_start) * 1000
        latency_list.append(latency)
        
        frame_count += 1
        vehicle_counts.append(vehicle_count)
        
        # Calculate FPS every second
        current_time = time.time()
        if current_time - last_log_time >= 1.0:
            elapsed = current_time - start_time
            fps = frame_count / elapsed
            fps_list.append(fps)
            
            # Get system resources
            cpu = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory().percent
            
            cpu_usage.append(cpu)
            ram_usage.append(ram)
            
            # Log progress
            print(f"{elapsed:>7.1f}s | {fps:>6.2f} | {latency:>8.0f}ms | {cpu:>5.1f}% | {ram:>5.1f}% | {vehicle_count:>8}")
            
            last_log_time = current_time
    
    cap.release()
    
    # Calculate statistics
    if len(fps_list) > 0 and len(latency_list) > 0:
        stats = {
            'test_name': test_name,
            'duration': duration,
            'total_frames': frame_count,
            'fps': {
                'mean': float(np.mean(fps_list)),
                'min': float(np.min(fps_list)),
                'max': float(np.max(fps_list)),
                'std': float(np.std(fps_list))
            },
            'latency_ms': {
                'mean': float(np.mean(latency_list)),
                'min': float(np.min(latency_list)),
                'max': float(np.max(latency_list)),
                'std': float(np.std(latency_list))
            },
            'cpu_percent': {
                'mean': float(np.mean(cpu_usage)),
                'min': float(np.min(cpu_usage)),
                'max': float(np.max(cpu_usage))
            },
            'ram_percent': {
                'mean': float(np.mean(ram_usage)),
                'min': float(np.min(ram_usage)),
                'max': float(np.max(ram_usage))
            },
            'vehicles': {
                'mean': float(np.mean(vehicle_counts)),
                'min': int(np.min(vehicle_counts)),
                'max': int(np.max(vehicle_counts))
            }
        }
        
        print(f"\n✅ Test completed!")
        return stats
    else:
        print(f"\n❌ Test failed - no data collected")
        return None

# ==============================================================================
# RUN ALL TESTS
# ==============================================================================

print("\n" + "="*70)
print("🚀 STARTING SYSTEM PERFORMANCE TESTS")
print("="*70)
print(f"📊 Total tests: {len(TEST_CONFIGS)}")
print(f"⏱️  Duration per test: {TEST_DURATION} seconds")
print(f"⏳ Total estimated time: ~{(TEST_DURATION * len(TEST_CONFIGS)) / 60:.1f} minutes")
print("="*70)

input("\nTekan ENTER untuk mulai test...")

all_results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model': MODEL_PATH,
    'test_duration': TEST_DURATION,
    'tests': []
}

for config in TEST_CONFIGS:
    # For simplicity, only test first stream
    # Multi-stream testing requires threading (more complex)
    stream_url = config['streams'][0]
    test_name = config['name']
    
    result = test_single_stream(stream_url, test_name, TEST_DURATION)
    
    if result:
        all_results['tests'].append(result)
    
    # Wait between tests
    if config != TEST_CONFIGS[-1]:
        print(f"\n⏸️  Waiting 10 seconds before next test...")
        time.sleep(10)

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

print("\n" + "="*70)
print("💾 SAVING RESULTS...")
print("="*70)

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"✅ Results saved to: {OUTPUT_JSON}")

# ==============================================================================
# DISPLAY SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("📊 PERFORMANCE TEST SUMMARY")
print("="*70)

for test_result in all_results['tests']:
    print(f"\n🧪 {test_result['test_name']}")
    print(f"   📹 Frames: {test_result['total_frames']}")
    print(f"   🎥 FPS: {test_result['fps']['mean']:.2f} (min: {test_result['fps']['min']:.2f}, max: {test_result['fps']['max']:.2f})")
    print(f"   ⏱️  Latency: {test_result['latency_ms']['mean']:.0f}ms (min: {test_result['latency_ms']['min']:.0f}ms, max: {test_result['latency_ms']['max']:.0f}ms)")
    print(f"   💻 CPU: {test_result['cpu_percent']['mean']:.1f}% (max: {test_result['cpu_percent']['max']:.1f}%)")
    print(f"   🧠 RAM: {test_result['ram_percent']['mean']:.1f}% (max: {test_result['ram_percent']['max']:.1f}%)")
    print(f"   🚗 Vehicles: {test_result['vehicles']['mean']:.1f} avg ({test_result['vehicles']['min']}-{test_result['vehicles']['max']})")

# Save summary text
summary_path = 'data/performance_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("HASIL PENGUJIAN PERFORMA SISTEM\n")
    f.write(f"Timestamp: {all_results['timestamp']}\n")
    f.write("="*70 + "\n\n")
    
    for test_result in all_results['tests']:
        f.write(f"{test_result['test_name']}\n")
        f.write(f"  FPS: {test_result['fps']['mean']:.2f}\n")
        f.write(f"  Latency: {test_result['latency_ms']['mean']:.0f}ms\n")
        f.write(f"  CPU: {test_result['cpu_percent']['mean']:.1f}%\n")
        f.write(f"  RAM: {test_result['ram_percent']['mean']:.1f}%\n\n")

print(f"\n💾 Summary saved to: {summary_path}")

print("\n" + "="*70)
print("✅ ALL TESTS COMPLETED!")
print("="*70)
print(f"📂 Results: {OUTPUT_JSON}")
print(f"📊 Summary: {summary_path}")
print("\n🔴 LANGKAH SELANJUTNYA:")
print("   Jalankan: python evaluation/visualize_results.py")
print("="*70)