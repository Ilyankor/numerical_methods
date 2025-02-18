# Gradient Descent with Python:

import numpy as np
from math import fsum

# For a specific Gradient Descent Problem
def f(U):
    return U[0]**2+U[0]*U[1]+3.0*U[1]**2+3.0*U[0]-4.0*U[1]
# The gradient of f
def df(U):
    return np.array([2.0*U[0]+U[1]+3.0,U[0]+6.0*U[1]-4.0])
# Gradient Descent with 100 iterations
def graddes(f,df,U0):
    # using 100 iterations
    n=100
    U=np.array(U0)
    m=0
    for i in range(n):
        s=1.0
        v=df(U)
        M=fsum(np.abs(v))
        # Exit if the gradient is too small.
        if M<10.0**(-12):
            return U
        fval=f(U)
        Unew=U-s/(M+1.0)*v
        while fval<f(Unew):
            m+=1
            s=s/2.0
            Unew=U-s/M*v
        U=Unew
    return U

# See Newton’s method on the next page…
# Newton’s Method with Python:
import numpy as np
# A function for Newton's method with g:R^n->R^n
def g(U):
    return np.array([4.*U[0]**3-3.*U[1]**2-3.,-6.*U[0]*U[1]+4.*U[1]**3+1.])
# The derivative is nxn matrix valued.
def dg(U):
    return np.array([[12.*U[0]**2,-6.*U[1]],[-6.*U[1],-6.*U[0]+12.*U[1]**2]])
def newton(g,dg,U0,tol):
    #g:R^n -> R^n and
    #dg(U0) should be an nxn invertible matrix
    #max of 10 iterations or until ||Unew - U|| < tol
    U=np.array(U0)
    for i in range(10):
        Unew=np.linalg.solve(dg(U),-g(U))+U
        if np.linalg.norm(Unew-U,2)<tol:
            return Unew
        U=Unew
    return U