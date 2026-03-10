import numpy as np

# Theoretical error bounds for the AFPM iterative refinement experiment.
# Mirrors the formulas in iterative-refinement/theoretical.py but assumes a
# single fixed precision (float32) for both the working and residual computation,
# since the AFPM hardware operates entirely in float32.

# return the unit roundoff for a given floating-point precision
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


# compute the componentwise condition number of the system Ax = b
def cond_Ax(A, x):
    A_inv = np.linalg.inv(A)
    abs_Ax = np.abs(A) @ np.abs(x)
    numerator = np.linalg.norm(np.abs(A_inv) @ abs_Ax, np.inf)
    denominator = np.linalg.norm(x, np.inf)
    return numerator / denominator


# calculate the theoretical convergence limit for the forward error
def calc_theoretical_limit(A, exact_x, p, precision="float32"):
    u = get_machine_epsilon(precision)
    u_r = u 

    c_cond = cond_Ax(A, exact_x)

    # Formula: Limit = 4 * p * u_r * cond(A, x) + u
    limit = 4 * p * u_r * c_cond + u

    return limit, u, u_r, c_cond


# count the maximum number of nonzeros in any row of A
def get_sparsity_p(A):
    non_zeros_per_row = np.count_nonzero(A, axis=1)
    p = np.max(non_zeros_per_row)
    return p