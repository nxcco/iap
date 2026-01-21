import numpy as np
from precisions import Precisions


def get_machine_epsilon(precision_str):
    """
    Returns the unit roundoff (machine epsilon) for the given precision type.

    Args:
        precision_str: String specifying precision (e.g., "float16", "float32", "float64")

    Returns:
        Machine epsilon (unit roundoff) for the specified precision
    """
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


def cond_Ax(A, x):
    """
    Compute the componentwise condition number cond(A, x).

    Formula: || |A^-1| * |A| * |x| ||_inf / ||x||_inf

    Args:
        A: Coefficient matrix
        x: Solution vector

    Returns:
        Componentwise condition number
    """
    # For small Poisson 1D problems, explicit inverse is acceptable for analysis
    A_inv = np.linalg.inv(A)

    # Numerator: || |A^-1| (|A| |x|) ||_inf
    abs_Ax = np.abs(A) @ np.abs(x)
    numerator = np.linalg.norm(np.abs(A_inv) @ abs_Ax, np.inf)

    # Denominator: ||x||_inf
    denominator = np.linalg.norm(x, np.inf)

    return numerator / denominator


def calc_theoretical_limit(A, exact_x, p):
    """
    Calculate the theoretical upper bound for forward error according to Corollary 3.3.

    Formula (3.10): Limit = 4 * p * u_r * cond(A, x) + u

    Args:
        A: Coefficient matrix
        exact_x: Exact solution vector
        p: Maximum number of non-zero elements per row (3 for 1D Poisson tridiagonal)

    Returns:
        Theoretical upper bound for relative forward error
    """
    # Get precision values from Precisions class
    u = get_machine_epsilon(Precisions.u)      # Working precision
    u_r = get_machine_epsilon(Precisions.u_r)  # Residual precision

    # Calculate condition number
    c_cond = cond_Ax(A, exact_x)

    # Apply formula (3.10): Limit = 4 * p * u_r * cond(A, x) + u
    limit = 4 * p * u_r * c_cond + u

    return limit, u, u_r, c_cond


def get_sparsity_p(A):
    """
    Determine the maximum number of non-zero elements per row in matrix A.

    Args:
        A: Coefficient matrix

    Returns:
        Maximum number of non-zero elements per row
    """
    # Count non-zero elements in each row
    non_zeros_per_row = np.count_nonzero(A, axis=1)
    p = np.max(non_zeros_per_row)
    return p


def check_convergence(c_cond):
    """
    Analyze and print convergence conditions for iterative refinement.

    Args:
        c_cond: Condition number cond(A,x)

    Returns:
        bool: True if the method is expected to converge (phi_s < 1)
    """
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
