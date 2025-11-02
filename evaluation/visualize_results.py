"""
Script untuk generate visualisasi hasil pengujian
Output: Grafik PNG untuk laporan
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ==============================================================================
# KONFIGURASI
# ==============================================================================

DETECTION_RESULTS = 'data/detection_results.csv'
PERFORMANCE_RESULTS = 'data/performance_results.json'
OUTPUT_DIR = 'results/figures'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# VISUALISASI 1: DETECTION ACCURACY
# ==============================================================================

print("="*70)
print("📊 GENERATING VISUALIZATIONS...")
print("="*70)

try:
    df_detection = pd.read_csv(DETECTION_RESULTS)
    
    print(f"\n✅ Loaded detection results: {len(df_detection)} frames")
    
    # Figure 1: Ground Truth vs Detected (Scatter Plot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Scatter plot
    axes[0, 0].scatter(df_detection['ground_truth'], df_detection['detected'], 
                       alpha=0.6, s=80, c='#3498db')
    max_val = max(df_detection['ground_truth'].max(), df_detection['detected'].max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Detection')
    axes[0, 0].set_xlabel('Ground Truth (Jumlah Aktual)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Detected (Jumlah Terdeteksi)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Perbandingan Ground Truth vs Detected', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error Distribution
    axes[0, 1].hist(df_detection['error'], bins=20, edgecolor='black', color='#e74c3c', alpha=0.7)
    axes[0, 1].axvline(x=0, color='green', linestyle='--', linewidth=2, label='No Error')
    axes[0, 1].set_xlabel('Error (Detected - Ground Truth)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Distribusi Error Deteksi', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: MAE by Waktu
    if 'waktu' in df_detection.columns:
        mae_by_time = df_detection.groupby('waktu')['abs_error'].mean().sort_values()
        colors = ['#2ecc71' if x < 2 else '#f39c12' if x < 4 else '#e74c3c' for x in mae_by_time.values]
        mae_by_time.plot(kind='bar', ax=axes[1, 0], color=colors, edgecolor='black')
        axes[1, 0].set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Waktu', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('MAE Berdasarkan Waktu', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(mae_by_time.values):
            axes[1, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Plot 4: Accuracy by Lokasi
    if 'lokasi' in df_detection.columns:
        acc_by_loc = df_detection.groupby('lokasi').apply(
            lambda x: 100 * (1 - x['abs_error'].sum() / x['ground_truth'].sum())
        ).sort_values(ascending=False)
        
        colors_acc = ['#2ecc71' if x >= 90 else '#f39c12' if x >= 80 else '#e74c3c' for x in acc_by_loc.values]
        acc_by_loc.plot(kind='bar', ax=axes[1, 1], color=colors_acc, edgecolor='black')
        axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Lokasi', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Akurasi Deteksi per Lokasi', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
        axes[1, 1].set_ylim([0, 105])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(acc_by_loc.values):
            axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    detection_viz_path = os.path.join(OUTPUT_DIR, 'detection_accuracy.png')
    plt.savefig(detection_viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {detection_viz_path}")
    plt.close()
    
except FileNotFoundError:
    print(f"⚠️  Detection results not found, skipping...")
except Exception as e:
    print(f"❌ Error generating detection visualizations: {e}")

# ==============================================================================
# VISUALISASI 2: PERFORMANCE METRICS
# ==============================================================================

try:
    with open(PERFORMANCE_RESULTS, 'r') as f:
        perf_data = json.load(f)
    
    print(f"✅ Loaded performance results: {len(perf_data['tests'])} tests")
    
    # Figure 2: Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    test_names = [t['test_name'] for t in perf_data['tests']]
    fps_means = [t['fps']['mean'] for t in perf_data['tests']]
    latency_means = [t['latency_ms']['mean'] for t in perf_data['tests']]
    cpu_means = [t['cpu_percent']['mean'] for t in perf_data['tests']]
    ram_means = [t['ram_percent']['mean'] for t in perf_data['tests']]
    
    # Plot 1: FPS Comparison
    axes[0, 0].bar(range(len(test_names)), fps_means, color='#3498db', edgecolor='black', alpha=0.8)
    axes[0, 0].set_xticks(range(len(test_names)))
    axes[0, 0].set_xticklabels(test_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('FPS (Frames Per Second)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Perbandingan FPS', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(fps_means):
        axes[0, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Plot 2: Latency Comparison
    axes[0, 1].bar(range(len(test_names)), latency_means, color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[0, 1].set_xticks(range(len(test_names)))
    axes[0, 1].set_xticklabels(test_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Latency (milliseconds)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Perbandingan Latency', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(latency_means):
        axes[0, 1].text(i, v + 5, f'{v:.0f}ms', ha='center', fontweight='bold')
    
    # Plot 3: CPU Usage
    axes[1, 0].bar(range(len(test_names)), cpu_means, color='#f39c12', edgecolor='black', alpha=0.8)
    axes[1, 0].set_xticks(range(len(test_names)))
    axes[1, 0].set_xticklabels(test_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('CPU Usage (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Penggunaan CPU', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(cpu_means):
        axes[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Plot 4: RAM Usage
    axes[1, 1].bar(range(len(test_names)), ram_means, color='#9b59b6', edgecolor='black', alpha=0.8)
    axes[1, 1].set_xticks(range(len(test_names)))
    axes[1, 1].set_xticklabels(test_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('RAM Usage (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Penggunaan RAM', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(ram_means):
        axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    performance_viz_path = os.path.join(OUTPUT_DIR, 'performance_comparison.png')
    plt.savefig(performance_viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {performance_viz_path}")
    plt.close()
    
except FileNotFoundError:
    print(f"⚠️  Performance results not found, skipping...")
except Exception as e:
    print(f"❌ Error generating performance visualizations: {e}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("✅ VISUALIZATION GENERATION COMPLETED!")
print("="*70)
print(f"📂 Output directory: {OUTPUT_DIR}")
print(f"📊 Files generated:")

for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith('.png'):
        print(f"   • {filename}")

print("\n🔴 LANGKAH SELANJUTNYA:")
print("   1. Buka folder results/figures/")
print("   2. Gunakan grafik untuk laporan PDF")
print("   3. Ambil screenshot dari web interface untuk routing test")
print("="*70)