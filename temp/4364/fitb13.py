from thomas import Thomas
from numpy import linspace, sin, cos, pi, sum, array, matmul, linalg
from rungekutta4 import rk4sys

# question 1

n = 100
x_0 = 0
x_1 = 1
a = 1
b = 2
h = (x_1-x_0)/n

def p(x):
    return x

def q(x):
    return 3*sin(pi*x)

def f(x):
    return -1 - cos(x)

x = linspace(0.01,0.99,99)
subdiag = [1 + 0.5*h*p(x) for x in x]
subdiag.pop(0)
superdiag = [1 - 0.5*h*p(x) for x in x]
superdiag.pop(n-2)
diag = [-2-h**2*q(x) for x in x]
fvals = [h**2*f(x) for x in x]
fvals[0] = h**2*f(0.01) - a*(1 + 0.5*h*p(0.01))
fvals[n-2] = h**2*f(0.99) - b*(1 - 0.5*h*p(0.99))

print("The answer to question 1 is:", diag[36])

# question 2

print("The answer to question 2 is:", superdiag[36])

# question 3

print("The answer to question 3 is:", subdiag[49])

# question 4

print("The answer to question 4 is:", fvals[46])

# question 5

print("The answer to question 5 is:", 99)

# question 6

print("The answer to question 6 is:", 99+98+98)

# question 7

print("The answer to question 7 is:", sum(fvals))

# question 8

print("The answer to question 8 is:", sum(subdiag) + sum(diag) + sum(superdiag))

# question 9

print("The answer to question 9 is:", Thomas(subdiag,diag,superdiag,fvals)[66])

# question 10 

def u(x,U):
    return [U[1],x*U[1] + 3*sin(pi*x)*U[0]]

def w(x,U):
    return [U[1],-1 - cos(x) + x*U[1] + 3*sin(pi*x)*U[0]]


A = array([
    [1,0,0,0],
    [0,1,0,0]
])

initial = array([
    [1],
    [2]
])

u_matrix = array([
    [1, 0],
    [rk4sys(u,0.01,100,0,[1,0])[0],     rk4sys(u,0.01,100,0,[0,1])[0]],
    [0,1],
    [rk4sys(u,0.01,100,0,[1,0])[1],     rk4sys(u,0.01,100,0,[0,1])[1]]
])

w_matrix = array([
    [0],
    [rk4sys(w,0.01,100,0,[0,0])[0]],
    [0],
    [rk4sys(w,0.01,100,0,[0,0])[1]]
])

M = matmul(A,u_matrix)
v = initial - matmul(A,w_matrix)
weights = matmul(linalg.inv(M),v)

print("The answer to question 10 is:", float(weights[0]*rk4sys(u,0.01,67,0,[1,0])[0] + weights[1]*rk4sys(u,0.01,67,0,[0,1])[0] + rk4sys(w,0.01,67,0,[0,0])[0]))