import numpy as np

# Builds the linear system Ax = b for the 1D Poisson problem on [0,1] with
# Dirichlet boundary conditions. Same formulation as in the iterative-refinement
# subproject, adapted here for use with approximate (AFPM) arithmetic.

# discretize the 1D Poisson problem on [0,1] with Dirichlet boundaries
def set_up_problem(N, f_val, u0, u1):
    A = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    
    h = 1.0 / (N + 1)
    
    b = np.full(N, f_val * h**2)
    b[0] += u0 
    b[-1] += u1
    
    return A, b

# compute the exact solution to the 1D Poisson problem
def get_exact_solution(N, f_val, u0, u1):
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1 - h, N)

    C2 = u0
    C1 = (u1 - u0 - f_val * (1 - 0) / 2)

    u_exact = -f_val * x**2 / 2 + C1 * x + C2

    return u_exact