import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython = True)
def RKHG(f,y0,t,D,args=()):
    n = len(t)
    size = len(y0)
    y = np.zeros((n, size))
    y[0] = y0
    h = t[1] - t[0]
    sh = np.sqrt(h)
    DD = np.sqrt(2*D)
    for i in range(n - 1):
        S = np.random.choice(np.array([-1,1]),size=size)
        dW = np.random.normal(0,1,size)*sh
        k1 = h*f(y[i],t[i],*args) + (dW - S*sh)*DD
        k2 = h*f(y[i]+k1,t[i]+h,*args) + (dW + S*sh)*DD
        y[i+1] = y[i] + 0.5*(k1+k2)
    return y
