import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import struct
import argparse
import yaml
from tqdm import tqdm
import _import_config

# Pfad-Hack für Module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FPM_T6Mx_PRIMR import *

# --- Config Laden ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config_primr8.yaml')
with open(CONFIG_PATH, 'r') as f:
    full_yaml = yaml.safe_load(f)

if 'afpm_analysis' in full_yaml and 'configurations' in full_yaml['afpm_analysis']:
    ALL_CONFIGS = full_yaml['afpm_analysis']['configurations']
else:
    ALL_CONFIGS = full_yaml

# Nur FPM_T6... Configs filtern
CHROMOSOMES = {k: v['chromosome'] for k, v in ALL_CONFIGS.items() if 'chromosome' in v}

# --- Hilfsfunktionen ---
def float_to_32bit_binary(f):
    packed = struct.pack('>f', f)
    bits = struct.unpack('>I', packed)[0]
    binary = [0] * 32
    for i in range(32):
        binary[31-i] = (bits >> i) & 1
    return binary

def binary_to_float(binary):
    bits = 0
    for i in range(32):
        if binary[31-i]:
            bits |= (1 << i)
    packed = struct.pack('>I', bits)
    return struct.unpack('>f', packed)[0]

def afpm_multiply(a, b, chromosome):
    # Optimierung: Wenn Chromosom trivial ist (exact)
    if all(c == 0 for c in chromosome):
         return float(a) * float(b)
         
    a_binary = float_to_32bit_binary(float(a))
    b_binary = float_to_32bit_binary(float(b))
    result_binary = AFPM(a_binary, b_binary, chromosome)
    return binary_to_float(result_binary)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config Name (e.g. FPM_T6_P41)')
    parser.add_argument('--samples', type=int, default=2_000_000)
    args = parser.parse_args()
    
    config_name = args.config
    # Falls User "P41" statt "FPM_T6_P41" eingibt, versuchen wir zu fixen
    if config_name not in CHROMOSOMES:
        candidate = f"FPM_T6_{config_name}"
        if candidate in CHROMOSOMES:
            config_name = candidate
        else:
            print(f"Error: Config {config_name} not found.")
            sys.exit(1)

    chromosome = CHROMOSOMES[config_name]
    N = args.samples
    
    # Short name for display
    short_name = config_name.replace("FPM_T6_", "")
    
    print(f"--- Plotting Histogram: {short_name} ({N/1e6:.1f}M Samples) ---")
    
    # 1. Daten generieren
    # Wir nutzen float32 für Speed und RAM
    A = np.random.uniform(-1000, 1000, N).astype(np.float32)
    B = np.random.uniform(-1000, 1000, N).astype(np.float32)
    rel_errors = np.zeros(N, dtype=np.float32)
    
    chunk_size = 50000
    num_chunks = N // chunk_size
    idx = 0
    
    for _ in tqdm(range(num_chunks), desc=f"Simulating {short_name}"):
        for _ in range(chunk_size):
            a_val = A[idx]
            b_val = B[idx]
            idx += 1
            
            exact = float(a_val) * float(b_val)
            if abs(exact) < 1e-12: continue
            
            approx = afpm_multiply(a_val, b_val, chromosome)
            rel_errors[idx-1] = (approx - exact) / exact
            
    # 2. Plot erstellen
    mean_bias = np.mean(rel_errors)
    std_dev = np.std(rel_errors)
    
    # Figure erstellen und explizit schließen später
    fig = plt.figure(figsize=(10, 6))
    
    # Histogramm
    n, bins, patches = plt.hist(rel_errors * 100, bins=100, density=True, 
                                color='skyblue', edgecolor='black', alpha=0.7, label='Error Distribution')
    
    # Linien
    plt.axvline(0, color='green', linestyle='-', linewidth=2, label='Exact (0.0%)')
    plt.axvline(float(mean_bias * 100), color='red', linestyle='--', linewidth=2,
                label=f'Mean Bias ({mean_bias*100:.5f}%)')
    
    # Titel & Labels
    plt.title(f'Relative Error Distribution: {short_name}\n(N={N/1e6:.1f}M, Bias={mean_bias*100:.6f}%)')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Zusatzinfo im Plot
    stats_text = f"Std Dev: {std_dev*100:.6f}%\nMin: {np.min(rel_errors)*100:.5f}%\nMax: {np.max(rel_errors)*100:.5f}%"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    outfile = os.path.join(os.path.dirname(__file__), f'hist_{short_name}.png')
    plt.savefig(outfile, dpi=150)
    plt.close(fig) # Wichtig für Speicherfreigabe im Loop
    
    print(f"Saved plot to: {outfile}")
    
    # Metadaten in kleine CSV append
    summary_file = os.path.join(os.path.dirname(__file__), '../results/summary_results.csv')
    exists = os.path.exists(summary_file)
    with open(summary_file, 'a') as f:
        if not exists:
            f.write("Config,MeanBias_Rel,StdDev,MinError_Rel,MaxError_Rel\n")
        f.write(f"{short_name},{mean_bias},{std_dev},{np.min(rel_errors)},{np.max(rel_errors)}\n")

if __name__ == "__main__":
    run()