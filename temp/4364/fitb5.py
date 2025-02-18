from integration import midpoint, trapezoidal, romberg, simpsons
from bisection import bisect
from numpy import sqrt, tan, zeros, cos
from math import fabs

# question 1

def f(x):
    return tan(sqrt(x**2+1))

print("The answer to question 1 is:", midpoint(f,100,0,1.1))

# question 2

print("The answer to question 2 is:", trapezoidal(f,100,0,1.1))

# question 3

def richardsonmidpoint(f,n,a,b,k):
    d = zeros((k,k))
    for i in range(k):
        d[i,0] = midpoint(f,n,a,b)
        for j in range(1,i+1):
            d[i,j] = d[i,j-1] + 1/(4**j-1)*(d[i,j-1] - d[i-1,j-1])
        n = 2*n
    return d[k-1,k-1]

print("The answer to question 3 is:", richardsonmidpoint(f,100,0,1.1,2))

# question 4

def richardsontrapezoidal(f,n,a,b,k):
    d = zeros((k,k))
    for i in range(k):
        d[i,0] = trapezoidal(f,n,a,b)
        for j in range(1,i+1):
            d[i,j] = d[i,j-1] + 1/(4**j-1)*(d[i,j-1] - d[i-1,j-1])
        n = 2*n
    return d[k-1,k-1]

print("The answer to question 4 is:", richardsontrapezoidal(f,100,0,1.1,2))

# question 5

print("The answer to question 5 is:", romberg(f,10**(-2),0,1.1))

# question 6

print("The answer to question 6 is:", simpsons(f,50,0,1.1))

# question 7

print("The answer to question 7 is:", romberg(f,10**(-1),0,1))

# question 8

print("The answer to question 8 is:", richardsonmidpoint(f,1,0,1,4))

# question 9

def romberg2(f,tol,a,b):
    n = 30  # max level is n
    rtable = zeros((n,n))
    rtable[0,0] = (f(a)+f(b))*(b-a)/2
    for i in range(n):
        rtable[i+1,0] = rtable[i,0]/2 + midpoint(f,2**i,a,b)/2
        for j in [k+1 for k in range(i+1)]:
            rtable[i+1-j,j] = rtable[i+2-j,j-1]+1/(4**(j)-1)*(rtable[i+2-j,j-1]-rtable[i+1-j,j-1])
        if fabs(rtable[0,i+1]-rtable[0,i]) < tol:
            return rtable[0,i+1]
    return print("tolerance not met")

def g(x):
    return romberg2(f,10**(-12),0,x) - 2

print("The answer to question 9 is:", bisect(g,0,1))

# question 10

def l(x):
    return x/(sqrt(x**2+1)*(cos(sqrt(x**2+1))**2))

def m(x):
    return sqrt(1+l(x)**2)

def o(x):
    return romberg2(m,10**(-12),0,x)-1

print("The answer to question 10 is:", bisect(o,0,1))