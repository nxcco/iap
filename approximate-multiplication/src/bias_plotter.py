import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import struct
import argparse
import _import_config
from FPM_T6Mx_PRIM import AFPM

# Pfad-Hack für Module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Hardcoded Configs (No YAML needed) ---
CHROMOSOMES = {
    'EXACT_32bit_MULTIPLY': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    'FPM_T6_P41': [100, 100, 100, 27, 27, 27, 0, 0, 0],
    'FPM_T6_P42': [100, 100, 100, 28, 28, 28, 0, 0, 0],
    'FPM_T6_P43': [100, 100, 100, 29, 29, 29, 0, 0, 0],
    'FPM_T6_P44': [100, 100, 100, 30, 30, 30, 0, 0, 0],
    'FPM_T6_P51': [100, 100, 100, 36, 36, 36, 0, 0, 0],
    'FPM_T6_P52': [100, 100, 100, 37, 37, 37, 0, 0, 0],
    'FPM_T6_P53': [100, 100, 100, 38, 38, 38, 0, 0, 0],
    'FPM_T6_P54': [100, 100, 100, 39, 39, 39, 0, 0, 0],
    'FPM_T6_P55': [100, 100, 100, 40, 40, 40, 0, 0, 0],
    'FPM_T6_P61': [100, 100, 100, 45, 45, 45, 0, 0, 0],
    'FPM_T6_P62': [100, 100, 100, 46, 46, 46, 0, 0, 0],
    'FPM_T6_P63': [100, 100, 100, 47, 47, 47, 0, 0, 0],
    'FPM_T6_P64': [100, 100, 100, 48, 48, 48, 0, 0, 0],
    'FPM_T6_P65': [100, 100, 100, 49, 49, 49, 0, 0, 0],
    'FPM_T6_P66': [100, 100, 100, 50, 50, 50, 0, 0, 0],
    'FPM_T6_P71': [100, 100, 100, 54, 54, 54, 0, 0, 0],
    'FPM_T6_P72': [100, 100, 100, 55, 55, 55, 0, 0, 0],
    'FPM_T6_P73': [100, 100, 100, 56, 56, 56, 0, 0, 0],
    'FPM_T6_P74': [100, 100, 100, 57, 57, 57, 0, 0, 0],
    'FPM_T6_P75': [100, 100, 100, 58, 58, 58, 0, 0, 0],
    'FPM_T6_P76': [100, 100, 100, 59, 59, 59, 0, 0, 0],
    'FPM_T6_P77': [100, 100, 100, 60, 60, 60, 0, 0, 0],
    'FPM_T6_P81': [100, 100, 100, 63, 63, 63, 0, 0, 0],
    'FPM_T6_P82': [100, 100, 100, 64, 64, 64, 0, 0, 0],
    'FPM_T6_P83': [100, 100, 100, 65, 65, 65, 0, 0, 0],
    'FPM_T6_P84': [100, 100, 100, 66, 66, 66, 0, 0, 0],
    'FPM_T6_P85': [100, 100, 100, 67, 67, 67, 0, 0, 0],
    'FPM_T6_P86': [100, 100, 100, 68, 68, 68, 0, 0, 0],
    'FPM_T6_P87': [100, 100, 100, 69, 69, 69, 0, 0, 0],
    'FPM_T6_P88': [100, 100, 100, 70, 70, 70, 0, 0, 0],
    'FPM_T6_P91': [100, 100, 100, 72, 72, 72, 0, 0, 0],
    'FPM_T6_P92': [100, 100, 100, 73, 73, 73, 0, 0, 0],
    'FPM_T6_P93': [100, 100, 100, 74, 74, 74, 0, 0, 0],
    'FPM_T6_P94': [100, 100, 100, 75, 75, 75, 0, 0, 0],
    'FPM_T6_P95': [100, 100, 100, 76, 76, 76, 0, 0, 0],
    'FPM_T6_P96': [100, 100, 100, 77, 77, 77, 0, 0, 0],
    'FPM_T6_P97': [100, 100, 100, 78, 78, 78, 0, 0, 0],
    'FPM_T6_P98': [100, 100, 100, 79, 79, 79, 0, 0, 0],
    'FPM_T6_P99': [100, 100, 100, 80, 80, 80, 0, 0, 0],
    'FPM_T6_Pa1': [100, 100, 100, 81, 81, 81, 0, 0, 0],
    'FPM_T6_Pa2': [100, 100, 100, 82, 82, 82, 0, 0, 0],
    'FPM_T6_Pa3': [100, 100, 100, 83, 83, 83, 0, 0, 0],
    'FPM_T6_Pa4': [100, 100, 100, 84, 84, 84, 0, 0, 0],
    'FPM_T6_Pa5': [100, 100, 100, 85, 85, 85, 0, 0, 0],
    'FPM_T6_Pa6': [100, 100, 100, 86, 86, 86, 0, 0, 0],
    'FPM_T6_Pa7': [100, 100, 100, 87, 87, 87, 0, 0, 0],
    'FPM_T6_Pa8': [100, 100, 100, 88, 88, 88, 0, 0, 0],
    'FPM_T6_Pa9': [100, 100, 100, 89, 89, 89, 0, 0, 0],
    'FPM_T6_Paa': [100, 100, 100, 99, 99, 99, 0, 0, 0],
}

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
    
    for _ in range(num_chunks):
        for _ in range(chunk_size):
            a_val = A[idx]
            b_val = B[idx]
            idx += 1
            
            reference = float(a_val) * float(b_val)
            if abs(reference) < 1e-12: continue
            
            approx = afpm_multiply(a_val, b_val, chromosome)
            rel_errors[idx-1] = (approx - reference) / reference
            
    # 2. Plot erstellen
    mean_bias = np.mean(rel_errors)
    std_dev = np.std(rel_errors)
    
    # Figure erstellen und explizit schließen später
    fig = plt.figure(figsize=(10, 6))
    
    # Histogramm
    n, bins, patches = plt.hist(rel_errors * 100, bins=100, density=True,
                                color='skyblue', edgecolor='black', alpha=0.7, label='Error Distribution')

    # Normalize x-axis for comparison across histograms
    # plt.xlim(-0.01, 0.00001)  # Fixed range: -0.001% to +0.0001%
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Linien
    plt.axvline(0, color='green', linestyle='-', linewidth=2, label='Reference (0.0%)')
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

if __name__ == "__main__":
    run()