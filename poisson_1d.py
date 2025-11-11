import numpy as np

# generate a SPD matrix from the 1D Poisson Problem
def setup_poisson_1d_problem(N, f_val, u0, u1):
    A = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    
    h = 1.0 / (N + 1)
    
    b = np.full(N, f_val * h**2)
    b[0] += u0 
    b[-1] += u1
    
    return A, b

# the analytical solution to the 1D poisson problem with contstant RHS
def exact_solution_1d_poisson(N, f_val, u0, u1):
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1 - h, N)

    C2 = u0
    C1 = (u1 - u0 - f_val * (1 - 0) / 2)

    u_exact = -f_val * x**2 / 2 + C1 * x + C2

    return u_exact