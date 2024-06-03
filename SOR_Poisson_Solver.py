#------------------------------------------
#Composite SOR solver
#------------------------------------------
L          = 1.0
ghost_zone = 1
N          = 128

def xy(N):
    x = np.linspace( 0.0, L, N+2*ghost_zone ) 
    y = np.linspace( 0.0, L, N+2*ghost_zone )
    return x,y

def source_func(N):
    exwhy = xy(N)
    rho = np.zeros((N+2*ghost_zone,N+2*ghost_zone))
    for i in range(N+2*ghost_zone):
        for j in range(N+2*ghost_zone):
            rho[i,j] = (-2*np.sin(exwhy[0][i])*np.sin(exwhy[1][j]))*(L/(N+1))**2 #multiplied by delta_x^2
    return rho

def u_ref(N):
    exwhy = xy(N)
    ref_func = np.zeros((N+2*ghost_zone,N+2*ghost_zone))
    for j in range(N+2*ghost_zone):
        for i in range(N+2*ghost_zone):
            ref_func[j,i] = np.sin(exwhy[0][i])*np.sin(exwhy[1][j])+1
    return ref_func

def difference(p1,p2):
    return np.abs(p1 - p2).sum()/np.abs(p2).sum()

def poisson_2d_SOR(p0, b, L, N, w, maxiter=20000, rtol=1e-6):
    p = p0.copy()
    conv = []  
    diff = rtol + 1.0  
    ite = 0  
    while diff > rtol and ite < maxiter:
        pn = p.copy()
        for i in range(1,N+1):
            for j in range(1,N+1):
                p[i,j] = (1-w)*p[i,j] + w*(p[i,j-1] + p[i,j+1] + p[i-1,j] + p[i+1,j] - b[i,j] * L/(N-1)**2) / 4
        diff = difference(p[1:-1, 1:-1],pn[1:-1, 1:-1])
        conv.append(diff)
        ite += 1
    return p, ite, conv

w_opt = 2/(1+np.sin(np.pi*(1/(N+1))))
def run_SOR(w, maxiter=20000, rtol=1e-10):
    '''
    w: The optimal omega, defined within the function, so you can just write w_opt
    Return: wall clock time, no. of iterations, error
    '''
    err = []
    wall_clock_time = []
    iterations = []
    x = np.linspace(0, 1, num=N)
    y = np.linspace(-0.5, 0.5, num=N)
    p0 = np.zeros((N+2*ghost_zone,N+2*ghost_zone))
    b = source_func(N)
    #Exact analytical solution
    p_exact = u_ref(N)
    
    start_time = time.time()
    p, ites, conv = poisson_2d_SOR(p0, b, L, N, w_opt, rtol=1e-10)
    end_time = time.time()
    diff = difference(p[1:-1, 1:-1],p_exact[1:-1, 1:-1])
    err.append(np.abs(diff-1))
    iterations.append(ites)
    wall_clock_time.append((end_time - start_time))
    return wall_clock_time, iterations, err

run_SOR(w_opt)

#Maybe something wrong with wall-clock time, please check