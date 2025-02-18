from firstorderdiffeq import *
import firstordersystem as fos
from numpy import cos, sin, pi
from newtons import newton

# question 1

def f(t,y):
    return 1+t-y**2

print('The answer to question 1 is:', euler(f,0.1,10,1,0.25))

# question 2

print('The answer to question 2 is:', midpoint(f,0.1,10,1,0.25))

# question 3

print('The answer to question 3 is:', modeuler(f,0.1,10,1,0.25))

# question 4

def j(t,U):
    return [U[1],1+cos(t)-0.1*U[1]+U[0]**2]

print('The answer to question 4 is:', fos.euler(j,0.1,10,0,[0.25,0.45]))

# question 5

print('The answer to question 5 is:', fos.midpoint(j,0.1,10,0,[0.25,0.45]))

# question 6

print('The answer to question 6 is:', fos.modeuler(j,0.1,10,0,[0.25,0.45]))

# question 7

g = 9.8
L = 4

def fosystem(t,U):
    return [U[1],-g/L*sin(U[0])]

def theta1(t):
    return fos.midpoint(fosystem,t/10000,10000,0,[pi/6,0])[0] - pi/6

def dtheta1(t):
    return fos.midpoint(fosystem,t/10000,10000,0,[pi/6,0])[1]

print("The answer to question 7 is:", newton(theta1,dtheta1,4,20))

# question 8

def theta2(t):
    return fos.midpoint(fosystem,t/10000,10000,0,[pi/4,0])[0] - pi/4

def dtheta2(t):
    return fos.midpoint(fosystem,t/10000,10000,0,[pi/4,0])[1]

print("The answer to question 8 is:", newton(theta2,dtheta2,4,20))


# question 9

def theta3(t):
    return fos.midpoint(fosystem,t/10000,10000,0,[pi/3,0])[0] - pi/3

def dtheta3(t):
    return fos.midpoint(fosystem,t/10000,10000,0,[pi/3,0])[1]

print("The answer to question 9 is:", newton(theta3,dtheta3,4,20))

# question 10

def theta4(t):
    return fos.midpoint(fosystem,t/10000,10000,0,[pi/2,0])[0] - pi/2

def dtheta4(t):
    return fos.midpoint(fosystem,t/10000,10000,0,[pi/2,0])[1]

print("The answer to question 10 is:", newton(theta4,dtheta4,4,20))


