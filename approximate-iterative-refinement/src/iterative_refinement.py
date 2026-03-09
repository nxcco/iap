import numpy as np
import cholesky
from afpm_utils import afpm_matvec

# Iterative refinement loop where both the residual computation and the correction
# solve use AFPM approximate multiplications, all in float32. This is the main
# experiment: can approximate hardware still drive the error down to the float32
# theoretical limit?

# Perform one refinement step: compute r = b - A*x using AFPM matrix-vector
# product, solve L L^T d = r with the pre-computed AFPM factorization, and return
# x + d. Everything stays in float32.
def refine(A, b, x, L):
    # Ensure inputs are float32
    A = A.astype(np.float32)
    b = b.astype(np.float32)
    x = x.astype(np.float32)

    # Compute residual using AFPM
    # r = b - A*x
    Ax = afpm_matvec(A, x) # Returns float32
    r = np.float32(b - Ax)

    # Solve for correction d using the pre-computed factorization L
    # Multiplications inside solve() also use AFPM
    d = cholesky.solve(L, r) # Returns float32

    # Update solution
    x_new = np.float32(x + d)

    return x_new

# Run up to max_iter AFPM refinement steps starting from an initial estimate x,
# using the pre-computed factorization L. Prints the residual (and optionally the
# forward error if x_exact is given) at each step. Returns the final solution.
def approximate(A, b, x, L, max_iter, x_exact=None):
    # Ensure inputs are float32
    A = A.astype(np.float32)
    b = b.astype(np.float32)
    x = x.astype(np.float32)
    
    for i in range(max_iter):
        x = refine(A, b, x, L)

        # Compute relative residual using AFPM
        Ax = afpm_matvec(A, x)
        residual = np.float32(b - Ax)
        
        # Norms are computed in standard precision (often float64 accumulator in numpy)
        # but inputs are float32. This is fine for metrics.
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(b)

        if x_exact is not None:
            # Compute relative forward error
            rel_error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
            print(f"  Iteration {i+1:3d}: Forward Error = {rel_error:.6e}, Residual = {rel_residual:.6e}")
        else:
            print(f"  Iteration {i+1:3d}: Residual = {rel_residual:.6e}")

    return x

# Full AFPM solver: factorize A once with AFPM Cholesky, get an initial solution,
# then call approximate() to refine it. The one-stop entry point when you just
# want a refined solution without managing the factorization yourself.
def solve(A, b, max_iter, x_exact=None):
    # Convert inputs to float32
    A = A.astype(np.float32)
    b = b.astype(np.float32)

    # Initial solve: factorize and solve using AFPM
    L = cholesky.factorize(A) # Returns float32 matrix
    x = cholesky.solve(L, b) # Returns float32 vector

    print(f"Starting approximate iterative refinement (max_iter={max_iter}, float32)...")
    if x_exact is not None:
        rel_error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
        print(f"  Initial:         Forward Error = {rel_error:.6e}")

    # Iteratively refine using the SAME factorization L
    x = approximate(A, b, x, L, max_iter=max_iter, x_exact=x_exact)

    return x