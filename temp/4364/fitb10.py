from rungekutta4 import rk4sys
from numpy import array, matmul, linalg, cos
from bisection import bisect

# question 1

def u(x,U):
    return [U[1],-2*U[1] + x*U[0]]

print("The answer to question 1 is: ", rk4sys(u,0.01,300,0,[1,0])[1])

# question 2

print("The answer to question 1 is: ", rk4sys(u,0.01,300,0,[0,1])[0])

# question 3

A1 = array([[1,0,-1,0],[0,2,0,1]])

Us1 = array([
    [1,0],
    [float(rk4sys(u,0.01,300,0,[1,0])[0]),float(rk4sys(u,0.01,300,0,[0,1])[0])],
    [0,1],
    [float(rk4sys(u,0.01,300,0,[1,0])[1]),float(rk4sys(u,0.01,300,0,[0,1])[1])]]
)

print("The answer to question 3 is: ", linalg.det(matmul(A1, Us1)))

# question 4

def w(x,U):
    return [U[1],-2*U[1] + x*U[0] + cos(x)]

Ws1 = array([
    [0],
    [float(rk4sys(w,0.01,300,0,[0,0])[0])],
    [0],
    [float(rk4sys(w,0.01,300,0,[0,0])[1])]]
)

v1 = array([[1],[2]]) - matmul(A1,Ws1)

print("The answer to question 4 is: ", matmul(linalg.inv(matmul(A1,Us1)), v1).item(0,0))

# question 5

print("The answer to question 5 is: ", matmul(linalg.inv(matmul(A1,Us1)), v1).item(1,0))

# question 6

print("The answer to question 6 is: ", matmul(linalg.inv(matmul(A1,Us1)), v1).item(0,0)*float(rk4sys(u,0.01,100,0,[1,0])[0]) + matmul(linalg.inv(matmul(A1,Us1)), v1).item(1,0)*float(rk4sys(u,0.01,100,0,[0,1])[0]) + float(rk4sys(w,0.01,100,0,[0,0])[0]))

# question 7

A2 = array([[0,0,1,0],[0,3,0,1]])
Us2 = array([
    [1,0],
    [float(rk4sys(u,0.01,200,-1,[1,0])[0]),float(rk4sys(u,0.01,200,-1,[0,1])[0])],
    [0,1],
    [float(rk4sys(u,0.01,200,-1,[1,0])[1]),float(rk4sys(u,0.01,200,-1,[0,1])[1])]]
)
Ws2 = array([
    [0],
    [float(rk4sys(w,0.01,200,-1,[0,0])[0])],
    [0],
    [float(rk4sys(w,0.01,200,-1,[0,0])[1])]]
)
v2 = array([[-1],[1]]) - matmul(A2,Ws2)
Cs2 = matmul(linalg.inv(matmul(A2,Us2)), v2)

print("The answer to question 7 is: ", Cs2.item(0,0)*float(rk4sys(u,0.01,100,-1,[1,0])[0]) + Cs2.item(1,0)*float(rk4sys(u,0.01,100,-1,[0,1])[0]) + float(rk4sys(w,0.01,100,-1,[0,0])[0]))

# question 8

def F1(x, U):
    return [U[1],U[0]**2 - x/(1+x)]

def temp1(x):
    return rk4sys(F1,0.01,100,0,[0,x])[0]

print("The answer to question 8 is: ", rk4sys(F1,0.01,50,0,[0,bisect(temp1,0,2)])[0])

# question 9

def F2(x, U):
    return [U[1], -1 - cos(x) - U[0] + U[0]**2]

def temp2(x):
    return rk4sys(F2,0.01,100,0,[x,0])[1]

value2_1 = rk4sys(F2,0.01,100,0,[bisect(temp2,-1,0),0])[0]
value2_2 = rk4sys(F2,0.01,100,0,[bisect(temp2,0,2),0])[0]

print("The answer to question 9 is: ", 0.5*(value2_1 + value2_2))

# question 10

def F3(x,U):
    return [U[1],U[0]**2-40*U[0]]

def temp3(x):
    return rk4sys(F3,0.01,100,0,[x,0])[0]

value3_1 = rk4sys(F3,0.01,50,0,[bisect(temp3,-18,-16),0])[0]
value3_2 = rk4sys(F3,0.01,50,0,[bisect(temp3,-1,1),0])[0]
value3_3 = rk4sys(F3,0.01,50,0,[bisect(temp3,35,37),0])[0]
value3_4 = rk4sys(F3,0.01,50,0,[bisect(temp3,38,40),0])[0]

print("The answer to question 10 is: ", 0.25*(value3_1+value3_2+value3_3+value3_4))