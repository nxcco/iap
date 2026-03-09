import numpy as np

# Builds the linear system Ax = b for the 1D Poisson problem on [0,1] with
# Dirichlet boundary conditions. Same formulation as in the iterative-refinement
# subproject, adapted here for use with approximate (AFPM) arithmetic.

# Discretise -u'' = f on [0,1] with u(0)=u0, u(1)=u1 using N interior points.
# Returns the tridiagonal SPD matrix A and the right-hand side b (with boundary
# values already folded in). The standard test bed for all solvers in this project.
def set_up_problem(N, f_val, u0, u1):
    A = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    
    h = 1.0 / (N + 1)
    
    b = np.full(N, f_val * h**2)
    b[0] += u0 
    b[-1] += u1
    
    return A, b

# Compute the closed-form solution to -u'' = f with constant f and Dirichlet
# boundaries at the same N interior grid points. Used as the reference "true"
# answer to measure forward error in the AFPM solver experiments.
def get_exact_solution(N, f_val, u0, u1):
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1 - h, N)

    C2 = u0
    C1 = (u1 - u0 - f_val * (1 - 0) / 2)

    u_exact = -f_val * x**2 / 2 + C1 * x + C2

    return u_exact