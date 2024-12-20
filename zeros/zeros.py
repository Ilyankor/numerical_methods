import numpy as np
from collections.abc import Callable

# bisection method
def bisection(func:Callable, a:float, b:float, tol:float=1e-12, n_max:int=1000) -> dict:

    # ensure that the interval is valid
    if func(a)*func(b) > 0:
        raise ValueError("The function may not change sign within the interval.")
    
    # ensure that a < b
    if a > b:
        a, b = b, a
    
    length = b - a # length of the interval
    
    for i in range(0, n_max+1):

        x = 0.5 * (a + b) # midpoint
        f_val = func(x) # function value at midpoint

        if f_val == 0: # check if the zero is found
            break

        # stopping criteria
        length *= 0.5
        if length < tol:
            break
       
        if f_val * func(a) < 0: # root is in [a, x]
            b = x
        else: # root is in [x, b]
            a = x

    # results
    result = {
        "zero": x,
        "iterations": i,
    }

    return result

def lazy_bisection(func:Callable, a:float, b:float, tol:float=1e-12, n_max:int=1000) -> dict:

    # ensure that the interval is valid
    if func(a)*func(b) > 0:
        raise ValueError("The function may not change sign within the interval.")
    
    # ensure that a < b
    if a > b:
        a, b = b, a
    
    length = b - a # length of the interval
    
    for i in range(0, n_max+1):

        x = 0.5 * (a + b) # midpoint
        f_val = func(x) # function value at midpoint

        # stopping criteria
        length *= 0.5
        if length < tol:
            break
       
        if f_val * func(a) <= 0: # root is in [a, x]
            b = x
        else: # root is in [x, b]
            a = x

    # results
    result = {
        "zero": x,
        "iterations": i,
    }

    return result


# chord method
# secant method
# regula falsi
# newton's method
# quasi Newton
# modified newton


# fixed point iteration
def fix_point(func:Callable, x0:float, tol:float, n_max:int, actual:float) -> tuple[float, float, float, list]:
    x = x0 # initial guess
    i = 0  # iteration counter
    abs_errors = [] # initialize list of absolute errors

    for i in range(1, n_max+1):
        x = func(x) # fixed point

        err = np.abs(x - func(x)) # error
        abs_errors.append(np.abs(x - actual)) # absolute error
        if err < tol:
            break
    
    # did not converge within n_max
    if err > tol:
        # sys.tracebacklimit = 0
        raise Exception(f"The fixed point iteration did not converge within {n_max} iterations.")

    return x, err, i, abs_errors

# define the bisection method
def bisect(f,a,b):
    p = (a+b)/2
    # check that a < b
    if a >= b:
        return print(a, ' is not less than ', b)
    # check that (a,b) is a bracket
    if f(a)*f(b) >= 0:
        return print('not a bracket')
    # iterate bisection until within 10^(-12) of a zero
    while abs(b-p) > 10**(-12):
        if f(p) == 0:
            return p
        elif f(a)*f(p) < 0:
            b = p
        else:
            a = p
        p = (a+b)/2
    return p

# find all the zeros
def findall(g,a,b,m):
    # [a,b] an interval, m a large positive integer.
    zerolist = []
    # check that a < b
    if a >= b:
        return print(a, ' is not less than ', b)
    # search intervals of (b-a)/m for bracketed roots.
    dx = (b-a)/m
    x0 = a
    for i in range(m):
        x1 = x0 + dx
        if g(x0)*g(x1) < 0:
            zerolist.append(bisect(g,x0,x1))
        x0 = x1
    return zerolist

p = np.array([2])

for i in range(6):
    p = np.append(p, p[i]/2 + 1/p[i])


# question 1b

from numpy import array, append

p = array([3])

for i in range(6):
    x = p[i]/2 + 3/(2*p[i])
    p = append(p,x)

print(p)

# # question 3

# from bisection import findall # automated bisection script
# from numpy import sin, cos, array, append

# def g(x):
#     return a*x**2+0.5*cos(5*x)+20*sin(x)

# p = array([])

# for i in range(1,11):
#     a = 0.001*i
#     p = append(p,len(findall(g,-1000,1000,10000)))

# print(p)



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
