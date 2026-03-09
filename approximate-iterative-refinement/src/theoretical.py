import numpy as np

def get_machine_epsilon(precision_str="float32"):
    """
    Returns the unit roundoff (machine epsilon) for the given precision type.
    """
    if precision_str == "float16":
        return 2**-11
    elif precision_str == "float32":
        return 2**-24
    elif precision_str == "float64":
        return 2**-53
    else:
        # Default to float32 if unknown or just return a reasonable default
        return 2**-24


def cond_Ax(A, x):
    """
    Compute the componentwise condition number cond(A, x).
    Formula: || |A^-1| * |A| * |x| ||_inf / ||x||_inf
    """
    A_inv = np.linalg.inv(A)
    abs_Ax = np.abs(A) @ np.abs(x)
    numerator = np.linalg.norm(np.abs(A_inv) @ abs_Ax, np.inf)
    denominator = np.linalg.norm(x, np.inf)
    return numerator / denominator


def calc_theoretical_limit(A, exact_x, p, precision="float32"):
    """
    Calculate the theoretical upper bound for forward error.
    Assuming fixed precision iterative refinement (u = u_r).
    """
    u = get_machine_epsilon(precision)
    u_r = u 

    c_cond = cond_Ax(A, exact_x)

    # Formula: Limit = 4 * p * u_r * cond(A, x) + u
    limit = 4 * p * u_r * c_cond + u

    return limit, u, u_r, c_cond


def get_sparsity_p(A):
    """
    Determine the maximum number of non-zero elements per row in matrix A.
    """
    non_zeros_per_row = np.count_nonzero(A, axis=1)
    p = np.max(non_zeros_per_row)
    return p