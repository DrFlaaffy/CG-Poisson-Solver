import numpy as np
import time

L          = 1.0
ghost_zone = 1
N          = 64
dimension  = N**2

def xy(N):
    x = np.linspace( 0.0, L, N+2*ghost_zone ) 
    y = np.linspace( 0.0, L, N+2*ghost_zone )
    return x,y

# define a reference analytical solution
def u_ref(N):
    exwhy = xy(N)
    ref_func = np.zeros((N+2*ghost_zone,N+2*ghost_zone))
    for j in range(N+2*ghost_zone):
        for i in range(N+2*ghost_zone):
            ref_func[j,i] = np.sin(exwhy[0][i])*np.sin(exwhy[1][j])+1
    return ref_func

#boundary condition
def source_func(N):
    exwhy = xy(N)
    rho = np.zeros((N+2*ghost_zone,N+2*ghost_zone))
    for j in range(N+2*ghost_zone):
        for i in range(N+2*ghost_zone):
            rho[j,i] = (-2*np.sin(exwhy[0][i])*np.sin(exwhy[1][j]))*(L/(N+1))**2
    return rho

def initial_bound(N):
    if ghost_zone > 1:
        raise ValueError("Too big")
    ib = source_func(N)
    ur = u_ref(N)
    ib[ghost_zone,ghost_zone:-ghost_zone] -=  ur[0,ghost_zone:-ghost_zone] #top
    ib[-ghost_zone-1,ghost_zone:-ghost_zone] -=  ur[-1,ghost_zone:-ghost_zone]#bottom
    ib[ghost_zone:-ghost_zone,ghost_zone] -=  ur[ghost_zone:-ghost_zone,0]#left
    ib[ghost_zone:-ghost_zone,-ghost_zone-1] -=  ur[ghost_zone:-ghost_zone,-1] #right
    
    return ib[ghost_zone:-ghost_zone,ghost_zone:-ghost_zone]

def A_matrix(idx_cell):# in range(dimension):
    idx_neighbor_list = []
    if (idx_cell%N  != 0 ):
        idx_neighbor_list.append(idx_cell-1) 
    if (idx_cell%N  != (N-1)):
        idx_neighbor_list.append(idx_cell+1)
    if (idx_cell//N != (N-1)):
        idx_neighbor_list.append(idx_cell+N)
    if (idx_cell//N != 0    ):
        idx_neighbor_list.append(idx_cell-N)
    return idx_neighbor_list

def Adotd(d): #dimension split is to split the dimension to different ranks, so it must be a list
    result = []
    for row in range(dimension):
        index_d = A_matrix(row)
        result.append(np.sum(d[index_d]) + d[row]*-4)
    return result

b = initial_bound(N).ravel()

x = np.zeros((N**2))  # first guess is zero vector
r = b  # r = b - A*x starts equal to b
d = r  # first search direction is r

start = time.time()
error = 100
while (error > 1e-16) == True:
    Ad = np.array(Adotd(d))
    alpha = np.dot(r.T, r) / np.dot(d.T, Ad)
    x = x + alpha * d  # step to next guess

    rnew = r - alpha * Ad  # update residual r
    beta = np.dot(rnew.T, rnew) / np.dot(r.T, r) #correction
    r = rnew
    d = r + beta * d  # compute new search direction
    error = np.dot(r.T,r)/N**4
 
print(time.time()-start, 's')
