import numpy as np
from numba import jit
@jit(nopython = True)
def get_Z(theta):
    z = np.mean(np.exp(1j*theta))
    return z

@jit(nopython = True)
def RKHG_Z(f,y0,t,D,args=()):
    n = len(t)
    size = len(y0)
    y0 = y0.copy()
    h = t[1] - t[0]
    sh = np.sqrt(h)
    zs = np.zeros(n,dtype=np.complex64)
    DD = np.sqrt(2*D)
    zs[0] = get_Z(y0)
    for i in range(n - 1):
        S = np.random.choice(np.array([-1,1]),size=size)
        dW = np.random.normal(0,sh,size)
        k1 = h*f(y0,t[i],*args) + (dW - S*sh)*DD
        k2 = h*f(y0+k1,t[i]+h,*args) + (dW + S*sh)*DD
        y0 = y0 + 0.5*(k1+k2)
        zs[i+1] = get_Z(y0)
    return zs
@jit(nopython = True)
def Kuramoto_AR(Theta,t,omega,N,K,mk,Aij,b):
    # print("Case m = 0")
    Theta = Theta.copy()
    theta = Theta[:N]
    theta_i= theta.reshape(1,-1)
    theta_j = theta_i.T
    # theta_i,theta_j = np.meshgrid(theta,theta,sparse=True)
    dtheta = omega +  K/mk*np.sum(Aij*np.sin(theta_j - theta_i),axis=0)  - b*np.sin(theta)
    return dtheta

@jit(nopython = True)
def ERKI_Z(f,y0,t,D,args=()):
    n = len(t)
    size = len(y0)
    y0 = y0.copy()
    h = t[1] - t[0]
    sh = np.sqrt(h)
    zs = np.zeros(n,dtype=np.complex64)
    zs[0] = get_Z(y0)
    for i in range(n - 1):
        dW = np.random.normal(0,sh,size)
        a0 = h*f(y0,t[i],*args)
        a1 = h*f(y0+a0,t[i]+h,*args)
        y0 = y0 + 0.5*(a0+a1) + D*dW
        zs[i+1] = get_Z(y0)
    return zs