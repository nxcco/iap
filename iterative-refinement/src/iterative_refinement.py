# Implements the iterative refinement loop on top of the Cholesky solver.
# Each step computes the residual in high precision (u_r), solves for a correction
# in low precision (u_s), and updates the solution in working precision (u). Over
# several iterations this drives the forward error toward the theoretical limit.

import numpy as np
import cholesky
from precisions import Precisions
from casting import to_prec

# perform one step of iterative refinement
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

# run iterative refinement steps
def approximate(A, b, x, L, max_iter, x_exact=None):
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

# full solver using iterative refinement
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