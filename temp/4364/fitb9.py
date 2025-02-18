from numpy import sin, cos, exp
from firstorderdiffeq import euler, midpoint, modeuler
from firstordersystem import euler as eulersys, midpoint as midsys, modeuler as modeulersys
from taylor import taylor2, taylor3
from rungekutta4 import rk4, rk4sys


# question 1

def y(t):
    return 17.6*exp(-0.5*t) + 0.4*cos(t) + 0.8*sin(t) + 2

print("The answer to question 1 is:", y(5))

# question 2

def f(t,y):
    return 1 + cos(t) - 0.5*y

print("The answer to question 2 is:", euler(f,0.1,50,0,20))

# question 3

print("The answer to question 3 is:", modeuler(f,0.1,50,0,20))

# question 4

print("The answer to question 4 is:", midpoint(f,0.1,50,0,20))

# question 5

def ft(t,y):
    return -sin(t)

def fy(t,y):
    return -0.5

print("The answer to question 5 is:", taylor2(f,ft,fy,0.1,50,0,20))

# question 6

def fyy(t,y):
    return 0

def ftt(t,y):
    return -cos(t)

def fty(t,y):
    return 0

print("The answer to question 6 is:", taylor3(f,ft,fy,fty,fyy,ftt,0.1,50,0,20))

# question 7

print("The answer to question 7 is:", rk4(f,0.1,50,0,20))

# question 8

def F(t,U):
    return [U[1],U[0]**2 - 1 - cos(t) - 6.0*U[0]]

print("The answer to question 8 is:", eulersys(F,0.01,500,0,[0,1])[0])

# question 9

print("The answer to question 9 is:", eulersys(F,0.01,500,0,[0,1])[1])

# question 10

print("The answer to question 10 is:", modeulersys(F,0.01,500,0,[0,1])[0])

# question 11

print("The answer to question 11 is:", modeulersys(F,0.01,500,0,[0,1])[1])

# question 12

print("The answer to question 12 is:", midsys(F,0.01,500,0,[0,1])[0])

# question 13

print("The answer to question 13 is:", midsys(F,0.01,500,0,[0,1])[1])

# question 14

print("The answer to question 14 is:", rk4sys(F,0.01,500,0,[0,1])[0])

# question 15

print("The answer to question 15 is:", rk4sys(F,0.01,500,0,[0,1])[1])