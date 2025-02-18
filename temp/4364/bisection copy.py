# automate the bisection method

import numpy as np

# define the function g(x) = 0.25*x^3 - 3*x^2 - x + 8 + 20*sin(15x)
def g(x):
    return 1/4*x**3-3*x**2-x+8+20*np.sin(15*x)

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
