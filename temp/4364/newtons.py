# newton's method in python

# define the function g(x) = 0.25*x^3 - 3*x^2 - x + 8
def g(x):
    return 1/4*x**3-3*x**2-x+8

# define the derivatve g'(x) = 0.75*x^2 - 6x - 1
def dg(x):
    return 3/4*x**2-6*x-1

# newton's method
def newton(f,df,p,n):
    # n is the number of iterations, p the initial guess
    for i in range(n):
        if df(p) == 0:
            return print('zero derivative encountered at ', p)
        else:
            p = p - f(p)/df(p)
    return p
