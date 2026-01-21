import numpy as np
from casting import to_prec
from precisions import Precisions

# Factorize the matrix
def factorize(A):
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(0, n):
        for j in range(0, i+1):
            if (i == j):
                sum_val = 0
                for k in range(0, i):
                    prod = to_prec(L[i, k]**2, Precisions.u_f)
                    sum_val = to_prec(sum_val + prod, Precisions.u_f)

                # Cast A[i,i] to u_f BEFORE subtraction to ensure consistent precision
                A_ij = to_prec(A[i, i], Precisions.u_f)
                sqrt_arg = to_prec(A_ij - sum_val, Precisions.u_f)
                L[i, j] = to_prec(np.sqrt(sqrt_arg), Precisions.u_f)
            else:
                sum_val = 0
                for k in range(0, j):
                    prod = to_prec(L[i, k] * L[j, k], Precisions.u_f)
                    sum_val = to_prec(sum_val + prod, Precisions.u_f)

                # Cast A[i,j] to u_f BEFORE subtraction to ensure consistent precision
                A_ij = to_prec(A[i, j], Precisions.u_f)
                numerator = to_prec(A_ij - sum_val, Precisions.u_f)
                L[i, j] = to_prec(numerator / L[j, j], Precisions.u_f)

    return L

# solve the matrix using cholesky matrix
def solve(L, b):
    n = L.shape[0]
    R = np.transpose(L)

    # Forward substitution: Ly = b
    y = np.zeros(n)
    for t in range(0, n):
        s = 0
        for k in range(0, t):
            prod = to_prec(L[t, k] * y[k], Precisions.u_s)
            s = to_prec(s + prod, Precisions.u_s)
        # Cast b[t] to u_s BEFORE subtraction to ensure consistent precision
        b_t = to_prec(b[t], Precisions.u_s)
        numerator = to_prec(b_t - s, Precisions.u_s)
        y[t] = to_prec(numerator / L[t, t], Precisions.u_s)

    # Backward substitution: R^T x = y
    x = np.zeros(n)
    for t in range(n-1, -1, -1):
        s = 0
        for k in range(t+1, n):
            prod = to_prec(R[t, k] * x[k], Precisions.u_s)
            s = to_prec(s + prod, Precisions.u_s)
        # y[t] is already in u_s precision from forward substitution, but cast for consistency
        y_t = to_prec(y[t], Precisions.u_s)
        numerator = to_prec(y_t - s, Precisions.u_s)
        x[t] = to_prec(numerator / R[t, t], Precisions.u_s)

    return x