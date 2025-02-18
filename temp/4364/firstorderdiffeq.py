import numpy as np

def f(t,y):
    return 1+np.cos(t)-y**2

def euler(f,h,n,t0,y0):
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