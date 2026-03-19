import numpy as np
from numba import jit

@jit(nopython=True)
def dZ_dt(Zs, t, alpha, beta, eta1, eta2, omega=0.0):
    # Zs 배열에는 a1, a2의 실수부/허수부만 포함 (b는 불필요)
    a1real, a1imag, a2real, a2imag = Zs
    a1 = a1real + 1j * a1imag
    a2 = a2real + 1j * a2imag
    
    # [수정 1] Eq (25): Z_{1, sigma} = (1 - 2*eta_sigma) * a_sigma^*
    Z11 = (1 - 2 * eta1) * np.conj(a1)
    Z12 = (1 - 2 * eta2) * np.conj(a2)
    
    # [수정 2] Eq (10) 바탕의 H_sigma 전개식 단순화
    H1 = (Z11 + beta * Z12)**2
    H2 = (Z12 + beta * Z11)**2
    
    A = np.exp(1j * alpha)
    AS = np.exp(-1j * alpha)
    
    # [수정 3] Eq (17): Dimension reduction (b_sigma 제거 및 고유 진동수 omega 추가)
    da1 = -1j * omega * a1 + 0.5 * (np.conj(H1) / a1 * A - H1 * a1**3 * AS)
    da2 = -1j * omega * a2 + 0.5 * (np.conj(H2) / a2 * A - H2 * a2**3 * AS)
    
    return np.array([da1.real, da1.imag, da2.real, da2.imag])
@jit(nopython=True)
def RK4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n,len(y0)))
    y[0] = y0

    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2.0, t[i] + h / 2.0, *args)
        k3 = f(y[i] + k2 * h / 2.0, t[i] + h / 2.0, *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y
def get_RQ_OA(Q1, Q2, alpha, beta, eta1, eta2, omega=0.0, shift=0, t_end=5000, dt=0.1):
    # 초기 조건 설정 (Eq (27)에 따라 r = sqrt(Q))
    a1 = np.sqrt(Q1) * np.exp(0j)
    a2 =  np.sqrt(Q2)*np.exp(1j * shift)

    t = np.arange(0, t_end, dt)
    
    # RK4 연산 (a1, a2 2개의 복소수 변수만 추적)
    Zs = RK4(dZ_dt, np.array([a1.real, a1.imag, a2.real, a2.imag]), t, args=(alpha, beta, eta1, eta2, omega))
    
    # 시계열 결과를 복소수 배열로 변환
    a1s = Zs[:, 0] + 1j * Zs[:, 1]
    a2s = Zs[:, 2] + 1j * Zs[:, 3]

    # [수정 4] Eq (25) & Eq (26, 27) 기반의 R, Q 질서 매개변수 계산
    RZ1 = (1 - 2 * eta1) * np.conj(a1s)
    RZ2 = (1 - 2 * eta2) * np.conj(a2s)
    
    QZ1 = np.conj(a1s)**2
    QZ2 = np.conj(a2s)**2

    R1s = np.abs(RZ1)
    R2s = np.abs(RZ2)
    Q1s = np.abs(QZ1)
    Q2s = np.abs(QZ2)
    
    return R1s, R2s, Q1s, Q2s, t