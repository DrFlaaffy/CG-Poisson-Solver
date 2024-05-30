#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def poisson_source(x, y, Lx, Ly):
    X, Y = np.meshgrid(x, y)
    b = (-2 * np.pi / Lx * np.pi / Ly * np.sin(np.pi * X / Lx) * np.cos(np.pi * Y / Ly))
    return b

#Analytical solution
def poisson_solution(x, y, Lx, Ly):
    X, Y = np.meshgrid(x, y)
    p = np.sin(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)
    return p

#Difference in L2 norm
def difference(p1,p2):
    return np.abs(p1 - p2).sum()/np.abs(p2).sum()

def poisson_2d_SOR(p0, b, dx, dy,nx,ny, w, maxiter=20000, rtol=1e-6):
    p = p0.copy()
    conv = []  
    diff = rtol + 1.0  
    ite = 0  
    while diff > rtol and ite < maxiter:
        pn = p.copy()
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                p[i,j] = (1-w)*p[i,j] + w*(p[i,j-1] + p[i,j+1] + p[i-1,j] + p[i+1,j] - b[i,j] * dx * dy) / 4
        diff = difference(p[1:-1, 1:-1],pn[1:-1, 1:-1])
        conv.append(diff)
        ite += 1
    return p, ite, conv
#-------------------------
#Parameters
#-------------------------
nx = np.array([16,32,64])
ny = nx
xmin, xmax = 0.0, 1.0  
ymin, ymax = -0.5, 0.5  
Lx = (xmax - xmin)  
Ly = (ymax - ymin)  
dx = Lx / (nx - 1)  
#-------------------------


err = []
wall_clock_time = []
iterations = []
for i in range(len(nx)):
    x = np.linspace(xmin, xmax, num=nx[i])
    y = np.linspace(ymin, ymax, num=ny[i])
    p0 = np.zeros((nx[i],ny[i]))
    b = poisson_source(x, y, Lx, Ly)
    #Optimal omega from Yang & Gobbert(2009)
    w_opt = 2/(1+np.sin(np.pi*(1/(nx[i]+1))))
    #Exact analytical solution
    p_exact = poisson_solution(x, y, Lx, Ly)
    start_time = time.time()
    p, ites, conv = poisson_2d_SOR(p0, b, dx[i], dy[i], nx[i], ny[i],w_opt, rtol=1e-10)
    end_time = time.time()
    diff = difference(p[1:-1, 1:-1],p_exact[1:-1, 1:-1])
    err.append(diff)
    iterations.append(ites)
    wall_clock_time.append((end_time - start_time))

