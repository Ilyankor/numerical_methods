import numpy as np
from collections.abc import Callable

def euler(f:Callable[[float], float], h:float, n:int, t0:float, y0:float):
    y = y0
    t = t0
    for i in range(n):
        y = y+f(t,y)*h
        t = t+h
    return y

def midpoint(f,h,n,t0,y0):
    y = y0
    t = t0
    for i in range(n):
        z = y+f(t,y)*h/2
        y = y+f(t+h/2,z)*h
        t = t+h
    return y

def modeuler(f,h,n,t0,y0):
    y = y0
    t = t0
    for i in range(n):
        z = y+f(t,y)*h
        y = y+(f(t,y)+f(t+h,z))*h/2
        t = t+h
    return y

def rk4(f,h,n,t0,U0):
    U = U0
    t = t0
    for i in range(n):
        k1 = h*f(t,U)
        k2 = h*f(t + h/2,U + k1/2)
        k3 = h*f(t + h/2,U + k2/2)
        k4 = h*f(t + h,U + k3)
        U = U + (k1 + 2*k2 + 2*k3 + k4)/6
        t = t + h
    return U

def rk4sys(F,h,n,t0,U0):
    U = U0
    t = t0
    k = len(U)
    for i in range(n):
        k1 = [h*F(t,U)[j] for j in range(k)]
        Z = [U[j] +k1[j]/2 for j in range(k)]
        k2 = [h*F(t+h/2,Z)[j] for j in range(k)]
        Z = [U[j] + k2[j]/2 for j in range(k)]
        k3 = [h*F(t+h/2,Z)[j] for j in range(k)]
        Z = [U[j]+k3[j] for j in range(k)]
        k4 = [h*F(t+h,Z)[j] for j in range(k)]
        U = [U[j]+(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6 for j in range(k)]
        t = t + h
    return U

def taylor2(g,gt,gy,h,n,t0,y0):
    y = y0
    t = t0
    for i in range(n):
        y = y + g(t,y)*h + 0.5*(gt(t,y) + gy(t,y)*g(t,y))*h**2
        t = t + h
    return y

def taylor3(g,gt,gy,gty,gyy,gtt,h,n,t0,y0):
    y = y0
    t = t0
    for i in range(n):
        z1 = gtt(t,y) + 2.0*gty(t,y)*g(t,y) + gyy(t,y)*g(t,y)**2
        z2 = gy(t,y)*(gt(t,y) + gy(t,y)*g(t,y))
        y = y + g(t,y)*h + 0.5*(gt(t,y) + gy(t,y)*g(t,y))*h**2 + (z1+z2)*h**3/6.0
        t = t + h
    return y
