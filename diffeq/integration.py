import numpy as np
from collections.abc import Callable

# integral using the midpoint method
def midpoint(f:Callable[[float], float], a:float, b:float, n:int=1000, h:float=0.001) -> float:
    '''
    Midpoint method: function f, left endpoint a, right endpoint b, number of rectangles n, step size h
    specify n OR h
    '''
    # ensure that a < b
    if a > b:
        a, b = b, a
    
    h = (b-a)/n
    x = a+h/2
    s = 0
    for i in range(n):
        s = s + f(x)
        x = x + h
    return s*h

def trapezoidal(f:Callable[[float], float], a:float, b:float, n:int=1000, h:float=0.001) -> float:
    h = (b-a)/n
    x = a
    s = (f(a)+f(b))/2
    for i in range(n-1):
        x = x + h
        s = s + f(x)
    return s*h

def simpsons(f,n,a,b):
    return 1/3*trapezoidal(f,n,a,b) + 2/3*midpoint(f,n,a,b)

def romberg(f,tol,a,b):
    n = 30  # max level is n
    rtable = np.zeros((n,n))
    rtable[0,0] = (f(a)+f(b))*(b-a)/2
    for i in range(n):
        rtable[i+1,0] = rtable[i,0]/2 + midpoint(f,2**i,a,b)/2
        for j in [k+1 for k in range(i+1)]:
            rtable[i+1-j,j] = rtable[i+2-j,j-1]+1/(4**(j)-1)*(rtable[i+2-j,j-1]-rtable[i+1-j,j-1])
        if np.abs(rtable[0,i+1]-rtable[0,i]) < tol:
            return [rtable[0,i+1],i+2]
    return print("tolerance not met")

def richmidpoint(f,tol,a,b):
    n = 30 # max level is n
    rtable = np.zeros((n,n))
    rtable[0,0] = f((a+b)/2)*(b-a)
    for i in range(n):
        rtable[i+1,0] = midpoint(f,2**(i+1),a,b)
        for j in [k+1 for k in range(i+1)]:
            rtable[i+1-j,j] = rtable[i+2-j,j-1]+1/(4**(j)-1)*(rtable[i+2-j,j-1]-rtable[i+1-j,j-1])
        if np.abs(rtable[0,i+1]-rtable[0,i]) < tol:
            return [rtable[0,i+1],i+2]
    return print("tolerance not met")
