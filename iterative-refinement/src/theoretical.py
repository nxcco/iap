import numpy as np
from precisions import Precisions

# Theoretical bounds for the iterative refinement method based on Corollary 3.3.
# Provides machine epsilons for various precisions, the componentwise condition
# number, and the predicted forward-error limit — used to compare against
# the measured error curves in the plots.

# Return the unit roundoff (u = 2^-(mantissa bits)) for float16/32/64/128.
# This is the theoretical smallest rounding error for one operation in that
# precision. All the error bound formulas in this file are built on these values.
def get_machine_epsilon(precision_str):
    if precision_str == "float16":  # half precision
        return 2**-11  # approximately 4.88e-4
    elif precision_str == "float32":  # single precision
        return 2**-24  # approximately 5.96e-8
    elif precision_str == "float64":  # double precision
        return 2**-53  # approximately 1.11e-16
    elif precision_str == "float128":  # quad precision
        return 2**-113  # approximately 9.63e-35
    else:
        raise ValueError(f"Unknown precision: {precision_str}")


# Compute the componentwise condition number cond(A, x) = || |A^-1| |A| |x| ||_inf
# / ||x||_inf. Unlike the standard 2-norm condition number this version captures
# how cancellation affects each component individually. Used in the theoretical
# forward-error bound and to check whether refinement will converge.
def cond_Ax(A, x):
    # For small Poisson 1D problems, explicit inverse is acceptable for analysis
    A_inv = np.linalg.inv(A)

    # Numerator: || |A^-1| (|A| |x|) ||_inf
    abs_Ax = np.abs(A) @ np.abs(x)
    numerator = np.linalg.norm(np.abs(A_inv) @ abs_Ax, np.inf)

    # Denominator: ||x||_inf
    denominator = np.linalg.norm(x, np.inf)

    return numerator / denominator


# Apply formula (3.10) from Corollary 3.3 to get the theoretical ceiling on the
# relative forward error after iterative refinement converges:
#   limit = 4 * p * u_r * cond(A,x) + u
# where p is the row sparsity, u_r is the residual precision, and u is the working
# precision. The horizontal dashed line in the convergence plots shows this value.
def calc_theoretical_limit(A, exact_x, p):
    # Get precision values from Precisions class
    u = get_machine_epsilon(Precisions.u)      # Working precision
    u_r = get_machine_epsilon(Precisions.u_r)  # Residual precision

    # Calculate condition number
    c_cond = cond_Ax(A, exact_x)

    # Apply formula (3.10): Limit = 4 * p * u_r * cond(A, x) + u
    limit = 4 * p * u_r * c_cond + u

    return limit, u, u_r, c_cond


# Count the maximum number of nonzeros in any single row of A. For the 1D Poisson
# tridiagonal matrix this is always 3. Passed as p into the theoretical bound
# formula — a denser matrix would give a larger (looser) bound.
def get_sparsity_p(A):
    # Count non-zero elements in each row
    non_zeros_per_row = np.count_nonzero(A, axis=1)
    p = np.max(non_zeros_per_row)
    return p


# Print whether the chosen precisions are expected to make refinement converge.
# Convergence requires phi_s = cond(A,x) * u_s < 1. If it's >= 1 the corrections
# can grow rather than shrink and the method will diverge. Used as a sanity check
# at the start of each experiment run.
def check_convergence(c_cond):
    u_f = get_machine_epsilon(Precisions.u_f)
    u_s = get_machine_epsilon(Precisions.u_s)
    phi_f = c_cond * u_f
    phi_s = c_cond * u_s

    print(f"\nconvergence analysis:")
    print(f"  convergence factor φ_f = cond(A,x) × u_f = {phi_f:.4f}")
    print(f"  convergence factor φ_s = cond(A,x) × u_s = {phi_s:.4f}")

    converges = phi_s < 1.0
    if converges:
        print(f"  ✓ (φ_s = {phi_s:.4f} < 1)")
    else:
        print(f"  ✗ (φ_s = {phi_s:.4f} ≥ 1)")
    print()

    return converges
