import numpy as np
import matplotlib.pyplot as plt
from poisson_1d import setup_poisson_1d_problem
from iterative_refinement import linsolve
from custom_precision_cholesky import fp16



def main():
    matrix_sizes = [n*5 for n in range(1, 15)]
    f_val = 1.0
    u0 = 0.0
    u1 = 0.0
    max_iter = 3 

    errors = []
    residuals = []

    print(f"Testing different matrix sizes with {max_iter} refinement iterations:\n")

    for N in matrix_sizes:
        print(f"N = {N}:")

        A, b = setup_poisson_1d_problem(N, f_val, u0, u1)

        x_fp16 = linsolve(A, b, fl=fp16, max_iter=max_iter)

        # Solve in full precision for reference
        x_fp64 = np.linalg.solve(A, b)

        # relative error (fp16 vs fp64)
        abs_error = np.linalg.norm(x_fp16 - x_fp64, ord=np.inf)
        norm_fp64 = np.linalg.norm(x_fp64, ord=np.inf)
        rel_error = abs_error / norm_fp64
        errors.append(rel_error)

        # relative residual
        abs_residual = np.linalg.norm(b - A @ x_fp16, ord=np.inf)
        norm_b = np.linalg.norm(b, ord=np.inf)
        rel_residual = abs_residual / norm_b
        residuals.append(rel_residual)

        print(f"  Relative Error (||x_fp16 - x_fp64||_inf / ||x_fp64||_inf): {rel_error:.6e}")
        print(f"  Relative Residual (||b - Ax_fp16||_inf / ||b||_inf): {rel_residual:.6e}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # error plot
    ax1.plot(matrix_sizes, errors, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Matrix Size (N)', fontsize=12)
    ax1.set_ylabel('Relative Error', fontsize=12)
    ax1.set_title('Relative Error vs Matrix Size', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # residual plot
    ax2.plot(matrix_sizes, residuals, 's-', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel('Matrix Size (N)', fontsize=12)
    ax2.set_ylabel('Relative Residual', fontsize=12)
    ax2.set_title('Relative Residual vs Matrix Size', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('error_residual_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved.")

if __name__ == "__main__":
    main()
