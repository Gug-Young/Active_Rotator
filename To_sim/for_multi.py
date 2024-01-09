import To_sim.Kuramoto_AR as KU
from To_sim.solver import RKHG
import numpy as np

def get_r_sigma(b,theta_random,t,D,omega,N,K,mk,Aij):
    th = len(t)//2
    N = len(omega)
    sol = RKHG(KU.Kuramoto_AR,theta_random,t,D, args=(omega,N,K,mk,Aij,b))
    theta_s = sol[th:,:N]
    dtheta_s = sol[th:,N:2*N] 
    rabs = np.mean(np.exp(theta_s.T*1j),axis=0)
    r = np.abs(rabs)
    sigma_phi = np.mean(rabs)
    sigma = np.abs(sigma_phi)
    psi = np.abs(sigma_phi)
    return np.mean(r),sigma