import numpy as np
from pychop import LightChop
import pychop

pychop.backend('numpy')

# Specifications according to "Minifloats" (Wikipedia)
fp16 = LightChop(exp_bits=5, sig_bits=10)
fp8 = LightChop(exp_bits=5, sig_bits=2)
fp4 = LightChop(exp_bits=2, sig_bits=1)

def factorize(A, fl):
    n = A.shape[0]
    L = np.zeros_like(A)
    
    for i in range(0, n):
        for j in range(0, i+1):
            if (i == j):
                sum = 0
                for k in range(0, i):
                    sum = fl(sum + fl((L[i, k])**2))
                    
                L[i,j] = fl(np.sqrt(fl(A[i,i] - sum)))
            else:
                sum = 0
                for k in range(0, j):
                    sum = fl(sum + fl(L[i, k] * L[j, k]))
                    
                L[i,j] = fl(fl(A[i,j] - sum)/L[j,j])
    
    return L

def solve(L, b, fl):
    n = L.shape[0]
    R = np.transpose(L)

    y = np.zeros(n)
    for t in range(0, n):
        s = 0
        for k in range(0, t):
            s = fl(s + fl(L[t,k] * y[k]))
        y[t] = fl(fl((b[t] - s ))/L[t,t])
        
    x = np.zeros(n)
    for t in range(n-1, -1, -1):
        s = 0
        for k in range(t+1, n):
            s = fl(s + fl(R[t,k] * x[k]))
        x[t] = fl(fl((y[t] - s ))/R[t,t])
        
    return x