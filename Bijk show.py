from numba import jit
import numpy as np
import matplotlib.pyplot as plt
# @jit(nopython=True)
def make_Bijk(N):
    Bijk = np.ones((N,N,N)) - np.eye(N)
    for i in range(N): Bijk[i,i,:]=0;Bijk[i,:,i]=0
    return Bijk
N = 10
Nt = N *2

Aij = np.ones((Nt,Nt)) - np.eye(Nt)
nu = 0.7 # inter group interaction
mu = 1 - nu # group-by group interaction
Bijk = make_Bijk(Nt) * mu
B0 = np.where(Bijk==0)
Bijk[:N,:N,:N] = nu
Bijk[N:,N:,N:] = nu
Bijk[B0] = 0

Nu0 = np.where(Bijk == nu)
Mu0 = np.where(Bijk == mu)


Bijk_box = np.zeros_like(Bijk,dtype=bool)
Bijk_box[np.where(Bijk!=0)] = 1
alpha = 0.5
colors = np .empty(list(Bijk_box.shape) + [4], dtype=np.float32)
# colors[Nu0] = [1, 0, 0, alpha]  # red
# colors[Mu0] = [0, 1, 0, alpha]  # red
colors[:N,:N,:N] = [1,0,0,alpha]
colors[N:,N:,N:] = [1,0,0,alpha]

colors[:N,:N,N:] = [0,1,0,alpha]
colors[:N,N:,:N] = [0,1,0,alpha]
colors[N:,N:,:N] = [0,1,0,alpha]
colors[N:,:N,N:] = [0,1,0,alpha]

colors[:N,N:,N:] = [0,0,1,alpha]
colors[N:,:N,:N] = [0,0,1,alpha]
# colors[*B0] = [0,0,1,alpha]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
# Voxels is used to customizations of the
# sizes, positions and colors.
ax.voxels(Bijk_box, facecolors=colors)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')




ax.plot(np.nan,np.nan,np.nan,'o',label= r'$\nu$|111,222',color='green')
ax.plot(np.nan,np.nan,np.nan,'o',label= r'$\mu$|121,122,221,212',color='red')
ax.plot(np.nan,np.nan,np.nan,'o',label= r'$\xi$|122,211',color='blue')
ax.legend(fontsize=6)


plt.show()