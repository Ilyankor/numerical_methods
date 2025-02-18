import numpy as np

def F(t,U):
    return [U[1],-1.0-np.cos(t)+0.1*U[1]+U[0]**2]

def euler(F,h,n,t0,U0):
    U = U0
    t = t0
    k = len(U)
    for i in range(n):
        U = [U[j]+F(t,U)[j]*h for j in range(k)]
        t = t+h
    return U

def midpoint(F,h,n,t0,U0):
    U = U0
    t = t0
    k = len(U)
    for i in range(n):
        Z = [U[j]+F(t,U)[j]*h/2 for j in range(k)]
        U = [U[j]+F(t+h/2,Z)[j]*h for j in range(k)]
        t = t+h
    return U

def modeuler(F,h,n,t0,U0):
    U = U0
    t = t0
    k = len(U)
    for i in range(n):
        Z = [U[j]+F(t,U)[j]*h for j in range(k)]
        U = [U[j]+(F(t,U)[j]+F(t+h,Z)[j])*h/2 for j in range(k)]
        t = t+h
    return U
