import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import struct
import argparse
from tqdm import tqdm
import _import_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FPM_T6Mx_PRIMR import *

# --- Chromosomen Definitionen ---
CHROMOSOMES = {
    "P41": [100, 100, 100, 27, 27, 27, 0, 0, 0],
    "P61": [100, 100, 100, 45, 45, 45, 0, 0, 0],
    "P81": [100, 100, 100, 63, 63, 63, 0, 0, 0],
    "EXACT": [0, 0, 0, 0, 0, 0, 0, 0, 0]
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
    if all(c == 101 for c in chromosome) or all(c == 0 for c in chromosome):
         return float(a) * float(b)
    a_binary = float_to_32bit_binary(float(a))
    b_binary = float_to_32bit_binary(float(b))
    result_binary = AFPM(a_binary, b_binary, chromosome)
    return binary_to_float(result_binary)

def run():
    parser = argparse.ArgumentParser(description='Generate AFPM Error Heatmap')
    parser.add_argument('--config', type=str, default='P41', choices=CHROMOSOMES.keys(), help='Configuration Name')
    parser.add_argument('--res', type=int, default=1000, help='Resolution (NxN)')
    args = parser.parse_args()
    
    config_name = args.config
    chromosome = CHROMOSOMES[config_name]
    RES = args.res
    
    print(f"--- Experiment: Error Heatmap ({config_name}) ---")
    print(f"Resolution: {RES}x{RES} ({RES**2/1e6:.1f} M pixels)")
    
    RANGE_MIN = 1.0
    RANGE_MAX = 2.0
    
    X = np.linspace(RANGE_MIN, RANGE_MAX, RES)
    Y = np.linspace(RANGE_MIN, RANGE_MAX, RES)
    Z = np.zeros((RES, RES), dtype=np.float32) # float32 spart RAM bei 5k
    
    # Berechnung
    # tqdm deaktivieren wenn wir parallel laufen, sonst Chaos im Log?
    # Wir lassen es an, parallel managed output meistens.
    for i in tqdm(range(RES), desc=f"Calculating {config_name}"):
        for j in range(RES):
            a = X[j]
            b = Y[i]
            
            exact = a * b
            approx = afpm_multiply(a, b, chromosome)
            
            if abs(exact) > 1e-15:
                ratio = approx / exact
            else:
                ratio = 1.0
                
            Z[i, j] = ratio

    # Statistik & Skalierung
    min_val = np.min(Z)
    max_val = np.max(Z)
    mean_val = np.mean(Z)
    
    delta = max(abs(min_val - 1.0), abs(max_val - 1.0))
    if delta < 1e-9: delta = 1e-6
    
    vmin = 1.0 - delta
    vmax = 1.0 + delta
    
    print(f"\n[{config_name}] Stats:")
    print(f"  Min: {min_val:.8f}")
    print(f"  Max: {max_val:.8f}")
    print(f"  Mean:{mean_val:.8f} (Bias)")

    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(Z, extent=(RANGE_MIN, RANGE_MAX, RANGE_MIN, RANGE_MAX), origin='lower',
               cmap='seismic', vmin=float(vmin), vmax=float(vmax), interpolation='nearest')
    
    plt.colorbar(label='Ratio (Approx/Exact)', fraction=0.046, pad=0.04)
    plt.title(f'AFPM Error Heatmap: {config_name}\nRes: {RES}x{RES}, Bias: {mean_val-1.0:.2e}')
    plt.xlabel('Mantissa A')
    plt.ylabel('Mantissa B')
    
    outfile = os.path.join(os.path.dirname(__file__), f'heatmap_{config_name}_{RES}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"[{config_name}] Saved to: {outfile}")

if __name__ == "__main__":
    run()
