import sys
import os
import struct
import numpy as np

# Wrapper around the AFPM (Approximate Floating-Point Multiplier) hardware model
# from the HPC submodule. Provides scalar and matrix-vector multiply using the
# bit-level AFPM simulation, plus a registry of named chromosome configurations
# that control how aggressively the multiplier approximates.

# Add HPC directory to path to import FPM_T6Mx_PRIM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'HPC')))
try:
    from FPM_T6Mx_PRIM import AFPM
except ImportError:
    # Fallback if running from a different context
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from HPC.FPM_T6Mx_PRIM import AFPM

# --- Hardcoded Configs ---
CHROMOSOMES = {
    'EXACT': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    'P41': [100, 100, 100, 27, 27, 27, 0, 0, 0],
    'P42': [100, 100, 100, 28, 28, 28, 0, 0, 0],
    'P43': [100, 100, 100, 29, 29, 29, 0, 0, 0],
    'P44': [100, 100, 100, 30, 30, 30, 0, 0, 0],
    'P51': [100, 100, 100, 36, 36, 36, 0, 0, 0],
    'P52': [100, 100, 100, 37, 37, 37, 0, 0, 0],
    'P53': [100, 100, 100, 38, 38, 38, 0, 0, 0],
    'P54': [100, 100, 100, 39, 39, 39, 0, 0, 0],
    'P55': [100, 100, 100, 40, 40, 40, 0, 0, 0],
    'P61': [100, 100, 100, 45, 45, 45, 0, 0, 0],
    'P62': [100, 100, 100, 46, 46, 46, 0, 0, 0],
    'P63': [100, 100, 100, 47, 47, 47, 0, 0, 0],
    'P64': [100, 100, 100, 48, 48, 48, 0, 0, 0],
    'P65': [100, 100, 100, 49, 49, 49, 0, 0, 0],
    'P66': [100, 100, 100, 50, 50, 50, 0, 0, 0],
    'P71': [100, 100, 100, 54, 54, 54, 0, 0, 0],
    'P72': [100, 100, 100, 55, 55, 55, 0, 0, 0],
    'P73': [100, 100, 100, 56, 56, 56, 0, 0, 0],
    'P74': [100, 100, 100, 57, 57, 57, 0, 0, 0],
    'P75': [100, 100, 100, 58, 58, 58, 0, 0, 0],
    'P76': [100, 100, 100, 59, 59, 59, 0, 0, 0],
    'P77': [100, 100, 100, 60, 60, 60, 0, 0, 0],
    'P81': [100, 100, 100, 63, 63, 63, 0, 0, 0],
    'P82': [100, 100, 100, 64, 64, 64, 0, 0, 0],
    'P83': [100, 100, 100, 65, 65, 65, 0, 0, 0],
    'P84': [100, 100, 100, 66, 66, 66, 0, 0, 0],
    'P85': [100, 100, 100, 67, 67, 67, 0, 0, 0],
    'P86': [100, 100, 100, 68, 68, 68, 0, 0, 0],
    'P87': [100, 100, 100, 69, 69, 69, 0, 0, 0],
    'P88': [100, 100, 100, 70, 70, 70, 0, 0, 0],
    'P91': [100, 100, 100, 72, 72, 72, 0, 0, 0],
    'P92': [100, 100, 100, 73, 73, 73, 0, 0, 0],
    'P93': [100, 100, 100, 74, 74, 74, 0, 0, 0],
    'P94': [100, 100, 100, 75, 75, 75, 0, 0, 0],
    'P95': [100, 100, 100, 76, 76, 76, 0, 0, 0],
    'P96': [100, 100, 100, 77, 77, 77, 0, 0, 0],
    'P97': [100, 100, 100, 78, 78, 78, 0, 0, 0],
    'P98': [100, 100, 100, 79, 79, 79, 0, 0, 0],
    'P99': [100, 100, 100, 80, 80, 80, 0, 0, 0],
    'Pa1': [100, 100, 100, 81, 81, 81, 0, 0, 0],
    'Pa2': [100, 100, 100, 82, 82, 82, 0, 0, 0],
    'Pa3': [100, 100, 100, 83, 83, 83, 0, 0, 0],
    'Pa4': [100, 100, 100, 84, 84, 84, 0, 0, 0],
    'Pa5': [100, 100, 100, 85, 85, 85, 0, 0, 0],
    'Pa6': [100, 100, 100, 86, 86, 86, 0, 0, 0],
    'Pa7': [100, 100, 100, 87, 87, 87, 0, 0, 0],
    'Pa8': [100, 100, 100, 88, 88, 88, 0, 0, 0],
    'Pa9': [100, 100, 100, 89, 89, 89, 0, 0, 0],
    'Paa': [100, 100, 100, 99, 99, 99, 0, 0, 0],
}

# Convert a Python float to a list of 32 bits (MSB first) matching the IEEE 754
# single-precision bit pattern. Needed to feed values into the AFPM bit-level model.
def float_to_32bit_binary(f):
    packed = struct.pack('>f', f)
    bits = struct.unpack('>I', packed)[0]
    binary = [0] * 32
    for i in range(32):
        binary[31-i] = (bits >> i) & 1
    return binary

# Reconstruct a Python float from a 32-bit list (MSB first). Inverse of
# float_to_32bit_binary — used to turn the AFPM output bits back into a number.
def binary_to_float(binary):
    bits = 0
    for i in range(32):
        if binary[31-i]:
            bits |= (1 << i)
    packed = struct.pack('>I', bits)
    return struct.unpack('>f', packed)[0]

# Multiply a and b using the AFPM hardware model with the given chromosome config.
# If the chromosome is all zeros (EXACT), falls back to standard float32 multiply.
# This is the core hardware-simulation step: converts to bits, calls AFPM, converts back.
def afpm_multiply(a, b, chromosome):
    # If using 'EXACT' chromosome or all zeros, use standard multiplication (in float32)
    if all(c == 0 for c in chromosome):
         return np.float32(a) * np.float32(b)
         
    a_binary = float_to_32bit_binary(float(a))
    b_binary = float_to_32bit_binary(float(b))
    result_binary = AFPM(a_binary, b_binary, chromosome)
    
    # Ensure the result is strictly float32
    return np.float32(binary_to_float(result_binary))

# Global configuration variable (can be set by main.py)
CURRENT_CHROMOSOME = CHROMOSOMES['Paa']

# Select a chromosome config by name (e.g. 'Paa') to use for all subsequent
# AFPM multiplications. Call this once at the start of an experiment to choose
# how approximate the hardware should be.
def set_active_chromosome(name):
    global CURRENT_CHROMOSOME
    if name in CHROMOSOMES:
        CURRENT_CHROMOSOME = CHROMOSOMES[name]
    else:
        # Try finding it with FPM_T6_ prefix if short name failed
        long_name = f"FPM_T6_{name}"
        if long_name in CHROMOSOMES: 
            pass 
        raise ValueError(f"Chromosome config '{name}' not found.")
    print(f"AFPM: Active chromosome set to {name}")

# Convenience wrapper: multiply a and b with the currently active chromosome.
# This is what all the solver code calls so it doesn't need to pass the chromosome
# around explicitly.
def afpm_mul(a, b):
    return afpm_multiply(a, b, CURRENT_CHROMOSOME)

# Compute the matrix-vector product Ax using afpm_mul for every scalar multiplication.
# The result is a float32 vector. Used in the refinement loop to compute the
# residual r = b - Ax with approximate hardware arithmetic.
def afpm_matvec(A, x):
    n = A.shape[0]
    # Handle 1D vector x properly
    if x.ndim == 1:
        m = len(x)
    else:
        m = x.shape[0]
        
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        sum_val = np.float32(0.0)
        for j in range(m):
            prod = afpm_mul(A[i, j], x[j])
            sum_val = np.float32(sum_val + prod)
        res[i] = sum_val
    return res