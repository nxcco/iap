import os
import numpy as np
import matplotlib.pyplot as plt
import poisson_1d as Poisson1D
import cholesky
from iterative_refinement import refine
from casting import to_prec
from src.theoretical import calc_theoretical_limit, get_sparsity_p, get_machine_epsilon, check_convergence
from precisions import Precisions


# generates and saves the plot for a iterative refinement process.
def iterative_refinement_of_fixed_matrix(N, max_iter, output_dir):
    f_val = 1.0
    u0 = 0.0
    u1 = 0.0
    
    # Set up the Poisson problem
    A, b = Poisson1D.set_up_problem(N, f_val, u0, u1)

    # Solve in full precision for reference (exact solution)
    x_fp64 = np.linalg.solve(A, b)

    # Calculate theoretical bounds
    p = get_sparsity_p(A)  # For 1D Poisson tridiagonal, p = 3
    theo_limit, u, u_r, c_cond = calc_theoretical_limit(A, x_fp64, p)

    u_f = get_machine_epsilon(Precisions.u_f)
    u_s = get_machine_epsilon(Precisions.u_s)

    print(f"setup:")
    print(f"  matrix size: {N}")
    print(f"  condition number cond(A,x): {c_cond:.2e}")
    print(f"  working precision (u): {Precisions.u} (epsilon = {u:.2e})")
    print(f"  factorization precision (u_f): {Precisions.u_f} (epsilon = {u_f:.2e})")
    print(f"  solving precision (u_s): {Precisions.u_s} (epsilon = {u_s:.2e})")
    print(f"  residual precision (u_r): {Precisions.u_r} (epsilon = {u_r:.2e})")
    print(f"  theoretical upper bound: {theo_limit:.2e}")

    check_convergence(c_cond)

    errors = []
    residuals = []
    iterations = []

    # --- Step 1: Factorize Once ---
    print("Computing Cholesky Factorization...")
    L = cholesky.factorize(A)

    # --- Step 2: Initial Solve ---
    print("Computing Initial Solution...")
    x_curr = cholesky.solve(L, b)
    # Cast initial solution to working precision
    x_curr = to_prec(x_curr, Precisions.u)

    # Record Initial State (Iteration 0)
    abs_error = np.linalg.norm(x_curr - x_fp64)
    norm_fp64 = np.linalg.norm(x_fp64)
    rel_error = abs_error / norm_fp64
    
    abs_residual = np.linalg.norm(b - A @ x_curr)
    norm_b = np.linalg.norm(b)
    rel_residual = abs_residual / norm_b

    errors.append(rel_error)
    residuals.append(rel_residual)
    iterations.append(0)
    
    print(f"Initial: Forward Error = {rel_error:.6e}, Residual = {rel_residual:.6e}")

    # --- Step 3: Iterative Refinement Loop ---
    for i in range(1, max_iter + 1):
        x_curr = refine(A, b, x_curr, L)

        # forward error
        abs_error = np.linalg.norm(x_curr - x_fp64)
        rel_error = abs_error / norm_fp64
        errors.append(rel_error)

        # Relative residual
        abs_residual = np.linalg.norm(b - A @ x_curr)
        rel_residual = abs_residual / norm_b
        residuals.append(rel_residual)

        iterations.append(i)
        print(f"Iteration {i:3d}: Forward Error = {rel_error:.6e}, Residual = {rel_residual:.6e}")

    print(f"\nRefined with {max_iter} iterations.")

    # plotting stuff
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.semilogy(iterations, errors, 'r-x', linewidth=2, markersize=6,
                 label='Measured Forward Error')

    ax1.axhline(y=theo_limit, color='k', linestyle='-', linewidth=6,
                label=f'Theoretical Bound ({theo_limit:.1e})')

    ax1.axhline(y=u, color='lime', linestyle='-', linewidth=2,
                label=f'Working Precision u ({u:.1e})')

    ax1.set_xlabel('Refinement Step', fontsize=12)
    ax1.set_ylabel('Relative Forward Error', fontsize=12)
    ax1.set_title(f'Forward Error ($u_r={Precisions.u_r}, u={Precisions.u}, u_s={Precisions.u_s}, u_f={Precisions.u_f}$)',
                  fontsize=14)
    ax1.grid(True, which="both", linestyle='--', alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(range(0, max_iter + 1))

    # residual plot
    ax2.semilogy(iterations, residuals, 's-', color='orange', linewidth=2, markersize=6,
                 label='Relative Residual')
    ax2.set_xlabel('Refinement Step', fontsize=12)
    ax2.set_ylabel('Relative Residual', fontsize=12)
    ax2.set_title('Residual', fontsize=14)
    ax2.grid(True, which="both", linestyle='--', alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(range(0, max_iter + 1))

    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"convergence_N{N}_ur{Precisions.u_r}_u{Precisions.u}_us{Precisions.u_s}_uf{Precisions.u_f}__iter{max_iter}_optimized.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {filename}")
    # plt.show()

if __name__ == "__main__":
    N = 100
    max_iter = 15
    output_dir = '../results'

    configs = [
        {"u_r": "float64", "u": "float32", "u_s": "float16", "u_f": "float16"},
        {"u_r": "float32", "u": "float16", "u_s": "float16", "u_f": "float16"},
    ]

    for cfg in configs:
        Precisions.u_r = cfg["u_r"]
        Precisions.u   = cfg["u"]
        Precisions.u_s = cfg["u_s"]
        Precisions.u_f = cfg["u_f"]
        iterative_refinement_of_fixed_matrix(N, max_iter, output_dir)