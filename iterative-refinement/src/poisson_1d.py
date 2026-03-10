import numpy as np

# Builds the linear system Ax = b for the 1D Poisson problem on [0,1] with
# Dirichlet boundary conditions. The resulting matrix A is tridiagonal and
# symmetric positive definite, making it a good test case for Cholesky.

# set up the 1D Poisson problem
def set_up_problem(N, f_val, u0, u1):
    A = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    
    h = 1.0 / (N + 1)
    
    b = np.full(N, f_val * h**2)
    b[0] += u0 
    b[-1] += u1
    
    return A, b

# compute the exact solution for the 1D Poisson problem
def get_exact_solution(N, f_val, u0, u1):
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1 - h, N)

    C2 = u0
    C1 = (u1 - u0 - f_val * (1 - 0) / 2)

    u_exact = -f_val * x**2 / 2 + C1 * x + C2

    return u_exact