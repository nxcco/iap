# This file implements the refinement, but not the cholesky part.

import numpy as np
import cholesky
from precisions import Precisions
from casting import to_prec

# Computes the new x to find.
def refine(A, b, x, L):
    # Compute residual in higher precision (u_r)
    # Cast A, b, x to u_r precision BEFORE computing the residual
    A_high = to_prec(A, Precisions.u_r)
    b_high = to_prec(b, Precisions.u_r)
    x_high = to_prec(x, Precisions.u_r)
    r = b_high - A_high @ x_high

    # Solve for correction d in solving precision (u_s) using pre-computed L
    d = cholesky.solve(L, r)

    # Update solution in working precision (u)
    # Cast d to working precision before adding to ensure consistent precision
    d_work = to_prec(d, Precisions.u)
    x_new = x + d_work
    # x is already in u, d_work is now in u, so x_new should be in u automatically
    # But cast to be absolutely sure
    x_new = to_prec(x_new, Precisions.u)

    return x_new

# Call the refine function multiple times but does not compute the factorization.
def approximate(A, b, x, L, max_iter, x_exact=None):
    """
    Perform multiple iterations of iterative refinement using pre-computed factorization.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        x: Initial solution estimate
        L: Pre-computed Cholesky factorization of A
        max_iter: Maximum number of refinement iterations
        x_exact: Optional exact solution for error computation
        verbose: If True, print progress information

    Returns:
        Refined solution
    """
    for i in range(max_iter):
        x = refine(A, b, x, L)

       
        # Compute relative residual
        residual = b - A @ x
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(b)

        if x_exact is not None:
            # Compute relative forward error
            rel_error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
            print(f"  Iteration {i+1:3d}: Forward Error = {rel_error:.6e}, Residual = {rel_residual:.6e}")
        else:
            print(f"  Iteration {i+1:3d}: Residual = {rel_residual:.6e}")

    return x

# Solve linear system Ax = b using iterative refinement.
def solve(A, b, max_iter, x_exact=None):
    # Initial solve: factorize ONCE in u_f, solve in u_s
    L = cholesky.factorize(A)
    x = cholesky.solve(L, b)

    # Cast initial solution to working precision
    x = to_prec(x, Precisions.u)

    print(f"Starting iterative refinement (max_iter={max_iter})...")
    if x_exact is not None:
        rel_error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
        print(f"  Initial:         Forward Error = {rel_error:.6e}")

    # Iteratively refine using the SAME factorization L
    x = approximate(A, b, x, L, max_iter=max_iter, x_exact=x_exact)

    return x