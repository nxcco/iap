# Central place to configure the four precision levels used by the iterative
# refinement algorithm. Swap these strings to try different precision combinations
# without touching any of the algorithm code.

class Precisions:
    # Factorization Precision: Precision used for matrix factorization (e.g., Cholesky).
    u_f = "float16"  # Factorization needs to be accurate

    # Solving Precision: Precision used for solving the correction equation Ad = r.
    u_s = "float32" 
    
    # Working Precision: Base precision for input data (A, b) and solution (x).
    u = "float32"

    # Residual Precision: Precision used for calculating the residual r = b - Ax.
    u_r = "float64"

# ur < u < us < uf in terms of machine epsilon
