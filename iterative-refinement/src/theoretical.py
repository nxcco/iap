import numpy as np
from precisions import Precisions

# Theoretical bounds for the iterative refinement method based on Corollary 3.3.
# Provides machine epsilons for various precisions, the componentwise condition
# number, and the predicted forward-error limit — used to compare against
# the measured error curves in the plots.

# return the unit roundoff for a given precision
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


# compute the componentwise condition number
def cond_Ax(A, x):
    # For small Poisson 1D problems, explicit inverse is acceptable for analysis
    A_inv = np.linalg.inv(A)

    # Numerator: || |A^-1| (|A| |x|) ||_inf
    abs_Ax = np.abs(A) @ np.abs(x)
    numerator = np.linalg.norm(np.abs(A_inv) @ abs_Ax, np.inf)

    # Denominator: ||x||_inf
    denominator = np.linalg.norm(x, np.inf)

    return numerator / denominator


# calculate the theoretical forward error limit
def calc_theoretical_limit(A, exact_x, p):
    # Get precision values from Precisions class
    u = get_machine_epsilon(Precisions.u)      # Working precision
    u_r = get_machine_epsilon(Precisions.u_r)  # Residual precision

    # Calculate condition number
    c_cond = cond_Ax(A, exact_x)

    # Apply formula (3.10): Limit = 4 * p * u_r * cond(A, x) + u
    limit = 4 * p * u_r * c_cond + u

    return limit, u, u_r, c_cond


# get the maximum number of nonzeros in any row
def get_sparsity_p(A):
    # Count non-zero elements in each row
    non_zeros_per_row = np.count_nonzero(A, axis=1)
    p = np.max(non_zeros_per_row)
    return p


# check if the iterative refinement will converge
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
