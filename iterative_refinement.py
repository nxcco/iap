import numpy
from custom_precision_cholesky import *

def refine(A, b, x, fl):
    r = b - A@x
    L = factorize(A, fl)
    y = solve(L, r, fl)
    
    return x + y

def it_refine(A, b, x, fl, max_iter):
    for _ in range(max_iter):
        x = refine(A, b, x, fl)
    return x

def linsolve(A, b, fl, max_iter):
    x = solve(factorize(A, fl), b, fl)
    x = it_refine(A, b, x, fl, max_iter=max_iter)
    return x