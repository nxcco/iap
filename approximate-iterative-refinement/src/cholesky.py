import numpy as np
from afpm_utils import afpm_mul

# Cholesky factorization and triangular solve where every multiplication is
# done by the AFPM hardware model instead of exact IEEE arithmetic. All other
# operations (add, subtract, divide, sqrt) stay in float32. This lets us measure
# how approximate multiplications affect the accuracy of the factorization.

# compute the approximate Cholesky factorization of A
def factorize(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.float32)

    for i in range(0, n):
        for j in range(0, i+1):
            if (i == j):
                sum_val = np.float32(0.0)
                for k in range(0, i):
                    prod = afpm_mul(L[i, k], L[i, k])
                    sum_val = np.float32(sum_val + prod)
                
                # A[i,i] is float32
                A_ii = np.float32(A[i, i])
                diff = np.float32(A_ii - sum_val)
                L[i, j] = np.sqrt(max(np.float32(0.0), diff), dtype=np.float32)
            else:
                sum_val = np.float32(0.0)
                for k in range(0, j):
                    prod = afpm_mul(L[i, k], L[j, k])
                    sum_val = np.float32(sum_val + prod)

                A_ij = np.float32(A[i, j])
                numerator = np.float32(A_ij - sum_val)
                L[i, j] = np.float32(numerator / L[j, j])

    return L

# solve Ax = b using the approximate Cholesky factor L
def solve(L, b):
    n = L.shape[0]
    R = np.transpose(L) # L is already float32

    # Forward substitution: Ly = b
    y = np.zeros(n, dtype=np.float32)
    for t in range(0, n):
        s = np.float32(0.0)
        for k in range(0, t):
            prod = afpm_mul(L[t, k], y[k])
            s = np.float32(s + prod)
        
        b_t = np.float32(b[t])
        numerator = np.float32(b_t - s)
        y[t] = np.float32(numerator / L[t, t])

    # Backward substitution: R^T x = y
    x = np.zeros(n, dtype=np.float32)
    for t in range(n-1, -1, -1):
        s = np.float32(0.0)
        for k in range(t+1, n):
            prod = afpm_mul(R[t, k], x[k])
            s = np.float32(s + prod)
        
        y_t = np.float32(y[t])
        numerator = np.float32(y_t - s)
        x[t] = np.float32(numerator / R[t, t])

    return x