import numpy as np

def Thomas(c, d, e, F):
    #Returns the solution to Mx=F where M is square invertible tridiagonal 
    #matrix and F is a given vector. c, d and e are the subdiagonal, diagonal 
    #and superdiagonal of the matrix M. 
    k = len(d) #this is the number of equations
    cc=np.array([float(j) for j in c])
    dd=np.array([float(j) for j in d])
    ee=np.array([float(j) for j in e])
    FF=np.array([float(j) for j in F])
    u=np.array([float(j) for j in dd])
    for i in range(1, k): 
        m = cc[i-1]/dd[i-1]
        dd[i] = dd[i] - m*ee[i-1] 
        FF[i] = FF[i] - m*FF[i-1]
    u[k-1] = FF[k-1]/dd[k-1]
    for i in range(k-2, -1, -1):
        u[i] = (FF[i]-ee[i]*u[i+1])/dd[i]
    return u