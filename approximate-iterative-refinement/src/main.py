import os
import numpy as np
import matplotlib.pyplot as plt
import poisson_1d as Poisson1D
from iterative_refinement import refine
import cholesky
from afpm_utils import set_active_chromosome, afpm_matvec, CURRENT_CHROMOSOME
from theoretical import calc_theoretical_limit, get_sparsity_p

# Entry point for the AFPM iterative-refinement experiment. Sets up a 1D Poisson
# problem, runs the approximate refinement loop with the active AFPM chromosome,
# and produces two convergence plots: relative forward error and relative residual
# per iteration, both overlaid with the float32 theoretical bound.

# Solve an N×N Poisson system using AFPM-based Cholesky + iterative refinement,
# printing per-iteration metrics and saving side-by-side convergence plots to
# output_dir. The active AFPM chromosome must be set before calling this function.
def iterative_refinement_of_fixed_matrix(N, max_iter, output_dir):
    f_val = 1.0
    u0 = 0.0
    u1 = 0.0
    
    # Set up the Poisson problem
    A, b = Poisson1D.set_up_problem(N, f_val, u0, u1)

    # Solve in full precision (float64) for reference (Exact Solution)
    x_fp64 = np.linalg.solve(A, b)

    # Calculate theoretical limit for standard float32 (baseline)
    p = get_sparsity_p(A)
    theo_limit, u_eps, _, c_cond = calc_theoretical_limit(A, x_fp64, p, precision="float32")

    # Convert problem to float32 for the experiment
    A_f32 = A.astype(np.float32)
    b_f32 = b.astype(np.float32)

    print(f"--- Setup ---")
    print(f"Matrix size: {N}x{N}")
    print(f"Condition number: {c_cond:.2e}")
    print(f"Multiplication: Approximate (AFPM Paa)")
    print(f"Arithmetic: float32")
    print(f"Theoretical Limit (Standard float32): {theo_limit:.2e}")
    print(f"----------------")

    errors = []
    residuals = []
    iterations = []

    # --- Step 1: Factorize Once ---
    print("Computing Cholesky Factorization (AFPM)...")
    L = cholesky.factorize(A_f32)

    # --- Step 2: Initial Solve ---
    print("Computing Initial Solution (AFPM)...")
    x_curr = cholesky.solve(L, b_f32)

    # Record Initial State (Iteration 0)
    abs_error = np.linalg.norm(x_curr - x_fp64)
    norm_fp64 = np.linalg.norm(x_fp64)
    rel_error = abs_error / norm_fp64
    
    abs_residual = np.linalg.norm(b_f32 - afpm_matvec(A_f32, x_curr))
    norm_b = np.linalg.norm(b_f32)
    rel_residual = abs_residual / norm_b

    errors.append(rel_error)
    residuals.append(rel_residual)
    iterations.append(0)
    
    print(f"Initial: Forward Error = {rel_error:.6e}, Residual = {rel_residual:.6e}")

    # --- Step 3: Iterative Refinement Loop ---
    for i in range(1, max_iter + 1):
        # Perform ONE refinement step on the current x
        x_curr = refine(A_f32, b_f32, x_curr, L)

        # Compute Metrics
        abs_error = np.linalg.norm(x_curr - x_fp64)
        rel_error = abs_error / norm_fp64
        
        abs_residual = np.linalg.norm(b_f32 - afpm_matvec(A_f32, x_curr))
        rel_residual = abs_residual / norm_b

        errors.append(rel_error)
        residuals.append(rel_residual)
        iterations.append(i)

        print(f"Iteration {i:3d}: Forward Error = {rel_error:.6e}, Residual = {rel_residual:.6e}")

    print(f"\nCompleted {max_iter} iterations.")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 1. Forward Error Plot
    ax1.semilogy(iterations, errors, 'r-x', linewidth=2, markersize=8, label='Measured Forward Error')
    ax1.axhline(y=theo_limit, color='k', linestyle='--', linewidth=2, label=f'Theoretical Bound (std float32): {theo_limit:.1e}')
    ax1.axhline(y=u_eps, color='lime', linestyle=':', linewidth=2, label=f'float32 epsilon: {u_eps:.1e}')
    
    ax1.set_xlabel('Refinement Step', fontsize=12)
    ax1.set_ylabel('Relative Forward Error', fontsize=12)
    ax1.set_title(f'Forward Error Convergence (AFPM Paa, float32)', fontsize=14)
    ax1.grid(True, which="both", linestyle='--', alpha=0.3)
    ax1.legend(fontsize=10)

    # 2. Residual Plot
    ax2.semilogy(iterations, residuals, 's-', color='orange', linewidth=2, markersize=8, label='Relative Residual')
    ax2.set_xlabel('Refinement Step', fontsize=12)
    ax2.set_ylabel('Relative Residual', fontsize=12)
    ax2.set_title('Relative Residual vs Iteration Steps (float32)', fontsize=14)
    ax2.grid(True, which="both", linestyle='--', alpha=0.3)
    ax2.legend(fontsize=12)

    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"afpm_refinement_N{N}_iter{max_iter}_Paa_float32_theo_optimized.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {filename}")

if __name__ == "__main__":
    # Configuration
    set_active_chromosome('Paa')
    
    # Parameters
    N = 100
    max_iter = 15
    output_dir = '../results'

    iterative_refinement_of_fixed_matrix(N, max_iter, output_dir)
