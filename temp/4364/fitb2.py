from bisection import findall
from newtons import newton
from numpy import sin, cos, sum, array

# question 1

def f(x):
    return 3*x*sin(x)-1

def bisect_append(f,a,b):
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
        plist.append(p)
    return p

plist = []
bisect_append(f,-2,0)
print('The answer to question 1 is', plist[5])

# question 2

def g(x):
    return 0.01*x**2-x+20*cos(5*x)

print('The answer to question 2 is', sum(findall(g,-50,150,10000)))

# question 3

def h(x):
    return x**2-1+cos(x)

def dh(x):
    return 2*x-sin(x)

def ddh(x):
    return 2-cos(x)

def newton_modified(f,df,ddf,p,n):
    # n is the number of iterations, p the initial guess
    for i in range(n):
        if df(p) == 0:
            return print('zero derivative encountered at ', p)
        else:
            p = p - (f(p)*df(p))/(df(p)**2-f(p)*ddf(p))
    return p

print('The answer to question 3 is', newton_modified(h,dh,ddh,1,2))

# question 4

print('The answer to question 4 is', newton(dh,ddh,1,2))

# question 5

def false_position(f,a,b):
    p = a - (f(a)*(b-a))/(f(b)-f(a))
    # check that a < b
    if a >= b:
        return print(a, ' is not less than ', b)
    # check that (a,b) is a bracket
    if f(a)*f(b) >= 0:
        return print('not a bracket')
    # iterate bisection until within 10^(-12) of a zero
    while abs(b-p) > 10**(-14):
        if f(p) == 0:
            return p
        elif f(a)*f(p) < 0:
            b = p
        else:
            a = p
        p = a - (f(a)*(b-a))/(f(b)-f(a))
        plist.append(p)
    return p

plist = []
false_position(f,-2,0)
print('The answer to question 5 is', plist[7])

# question 8

def j(x):
    return 11*x-x**2+10*sin(x)

print('The answer to question 8 is', findall(j,-10,20,1000))

# question 9

def k(x):
    return 0.5*x**2 - x - 1 + 20*cos(10*x)

print('The answer to question 9 is', len(findall(k,-10,10,10000)))

# question 10

v = array(findall(k,-10,10,10000))
print('The answer to question 10 is', sum(v)/len(v))