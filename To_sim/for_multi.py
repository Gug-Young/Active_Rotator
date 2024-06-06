import To_sim.Kuramoto_AR as KU
from To_sim.solver import RKHG
from To_sim.Kuramoto_ARZ import RKHG_Z
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


def get_r_sigma_Z(b,theta_random,t,D,omega,N,K,mk,Aij):
    th = len(t)//2
    Zs = RKHG_Z(KU.Kuramoto_AR,theta_random,t,D, args=(omega,N,K,mk,Aij,b))
    r = np.abs(Zs[th:])
    r_m = np.mean(r)
    r_sm = np.mean(r**2)
    sigma_phi = np.mean(Zs[th:])
    sigma = np.abs(sigma_phi)
    psi = np.angle(sigma_phi)
    chi = (r_sm-sigma**2)*N
    return r_m,sigma,chi

def get_r_sigma_Z_D(D,theta_random,t,b,omega,N,K,mk,Aij):
    th = len(t)//2
    Zs = RKHG_Z(KU.Kuramoto_AR,theta_random,t,D, args=(omega,N,K,mk,Aij,b))
    r = np.abs(Zs[th:])
    r_m = np.mean(r)
    r_sm = np.mean(r**2)
    sigma_phi = np.mean(Zs[th:])
    sigma = np.abs(sigma_phi)
    psi = np.angle(sigma_phi)
    chi = (r_sm-sigma**2)*N
    return r_m,sigma,chi


def get_r_sigma_Z_MF(b,theta_random,t,D,omega,N,K):
    th = len(t)//2
    # Zs = RKHG_Z(KU.Kuramoto_mf_AR,theta_random,t,D, args=(omega,N,K,b))
    Zs = RKHG_Z(KU.Kuramoto_mf_AR,theta_random,t,D, args=(omega,N,K,b))
    r = np.abs(Zs[th:])
    r_m = np.mean(r)
    r_sm = np.mean(r**2)
    sigma_phi = np.mean(Zs[th:])
    sigma = np.abs(sigma_phi)
    psi = np.angle(sigma_phi)
    chi = (r_sm-sigma**2)*N
    return r_m,sigma,chi

def get_r_sigma_Z_MF_D(D,theta_random,t,b,omega,N,K):
    th = len(t)//2
    # Zs = RKHG_Z(KU.Kuramoto_mf_AR,theta_random,t,D, args=(omega,N,K,b))
    Zs = RKHG_Z(KU.Kuramoto_mf_AR,theta_random,t,D, args=(omega,N,K,b))
    r = np.abs(Zs[th:])
    r_m = np.mean(r)
    r_sm = np.mean(r**2)
    sigma_phi = np.mean(Zs[th:])
    sigma = np.abs(sigma_phi)
    psi = np.angle(sigma_phi)
    chi = (r_sm-sigma**2)*N
    return r_m,sigma,chi


def get_sol_MF(theta_random,t,D,b,omega,N,K):
    th = len(t)//2
    # Zs = RKHG_Z(KU.Kuramoto_mf_AR,theta_random,t,D, args=(omega,N,K,b))
    sol = RKHG(KU.Kuramoto_mf_AR,theta_random,t,D, args=(omega,N,K,b))
    theta_s = sol[:,:N]
    rabs = np.mean(np.exp(theta_s.T*1j),axis=0)
    r = np.abs(rabs)
    r_ = r[th:]
    r_m = np.mean(r)
    r_sm = np.mean(r**2)
    sigma_phi = np.mean(rabs[th:])
    sigma = np.abs(sigma_phi)
    psi = np.angle(sigma_phi)
    chi = (r_sm-sigma**2)*N
    return theta_s,rabs,chi,sigma_phi



def get_sol(theta_random,t,D,b,omega,N,K,mk,Aij):
    th = len(t)//2
    sol = RKHG(KU.Kuramoto_AR,theta_random,t,D, args=(omega,N,K,mk,Aij,b))
    theta_s = sol[:,:N]
    rabs = np.mean(np.exp(theta_s.T*1j),axis=0)
    r = np.abs(rabs)
    r_ = r[th:]
    r_m = np.mean(r)
    r_sm = np.mean(r**2)
    sigma_phi = np.mean(rabs[th:])
    sigma = np.abs(sigma_phi)
    psi = np.angle(sigma_phi)
    chi = (r_sm-sigma**2)*N
    return theta_s,rabs,chi,sigma_phi