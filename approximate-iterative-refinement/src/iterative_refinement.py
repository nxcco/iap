import numpy as np
import cholesky
from afpm_utils import afpm_matvec

# Computes the new x to find using one step of iterative refinement (in float32).
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

# Call the refine function multiple times
def approximate(A, b, x, L, max_iter, x_exact=None):
    """
    Perform multiple iterations of iterative refinement using AFPM.
    All vector ops are float32.
    """
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

# Solve linear system Ax = b using iterative refinement with AFPM.
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