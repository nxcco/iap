import numpy as np

# Theoretical error bounds for the AFPM iterative refinement experiment.
# Mirrors the formulas in iterative-refinement/theoretical.py but assumes a
# single fixed precision (float32) for both the working and residual computation,
# since the AFPM hardware operates entirely in float32.

# Return the unit roundoff for float16/32/64. Defaults to float32 since that is
# the only precision the AFPM hardware uses in this subproject.
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


# Compute the componentwise condition number cond(A,x) = || |A^-1| |A| |x| ||_inf
# / ||x||_inf. Captures how sensitive the solution is to perturbations component-
# by-component. Used in the theoretical bound and convergence check.
def cond_Ax(A, x):
    A_inv = np.linalg.inv(A)
    abs_Ax = np.abs(A) @ np.abs(x)
    numerator = np.linalg.norm(np.abs(A_inv) @ abs_Ax, np.inf)
    denominator = np.linalg.norm(x, np.inf)
    return numerator / denominator


# Apply the formula limit = 4 * p * u_r * cond(A,x) + u to get the floor the
# forward error should converge to. Because everything is float32 here, u = u_r.
# The result is the horizontal bound line drawn on the convergence plot.
def calc_theoretical_limit(A, exact_x, p, precision="float32"):
    u = get_machine_epsilon(precision)
    u_r = u 

    c_cond = cond_Ax(A, exact_x)

    # Formula: Limit = 4 * p * u_r * cond(A, x) + u
    limit = 4 * p * u_r * c_cond + u

    return limit, u, u_r, c_cond


# Count the maximum number of nonzeros in any row of A. For the 1D Poisson
# tridiagonal this is 3. Passed as p into the theoretical bound — a denser
# matrix would give a larger (looser) bound.
def get_sparsity_p(A):
    non_zeros_per_row = np.count_nonzero(A, axis=1)
    p = np.max(non_zeros_per_row)
    return p