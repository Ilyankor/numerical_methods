from bisection import bisect
from numpy import sin, cos, zeros
from math import pi,fabs

# question 1

def f(x):
    return sin(pi/4+0.1)-sin(pi/4)-0.1*cos(pi/4)+0.5*(0.1)**2*sin(x)

print("The answer to question 1 is: ", bisect(f, pi/4, pi/4+0.1))

# question 2

def g(x):
    return f(pi/4)+(1/6)*(0.1)**3*cos(x)

print("The answer to question 2 is: ", bisect(g, pi/4, pi/4+0.1))

# question 3

def h(x):
    return g(pi/4)-(1/24)*(0.1)**4*sin(x)

print("The answer to question 3 is: ", bisect(h, pi/4, pi/4+0.1))

# question 9 

def k(x):
    return 100000*sin(10*x)

def richardson(f,x,h,n):
    d = zeros((n,n))
    for i in range(n):
        d[i,0] = (f(x+h) - f(x-h))/(2*h)
        for j in range(1,i+1):
            d[i,j] = d[i,j-1] + 1/(4**j-1)*(d[i,j-1] - d[i-1,j-1])
        h = 0.5*h
    return d[n-1,n-1]

print('The answer to question 9 is:', richardson(k,1,0.1,4))

# question 10

diff = fabs(richardson(k,1,0.1,5) - 1000000*cos(10))*10**5

print('The answer to question 10 is:', diff)