from numpy import sin, cos, array, append, transpose, arange
from math import fabs
import matplotlib.pyplot as plt
from bisection import findall

# question 5

def f(x):
    return 2*x*sin(3*x)+3*x**2-2*x

h = array([0.5, 0.1, 0.05, 0.01, 0.005, 0.0001])
forward = array([])
backward = array([])
centered = array([])

for i in h:
    forward = append(forward, (f(i)-f(0))/i)
    backward = append(backward, (f(-i)-f(0))/(-i))
    centered = append(centered, (f(i)-f(-i))/(2*i))

values = transpose(array([forward, backward, centered]))

print('The answer to question 5 is:', values)

# question 6

def f1(x): # derivative
    return 2*sin(3*x)+6*x*cos(3*x)+6*x-2

def g1(x,h): # centered
    return (f(x+h)-f(x-h))/(2*h)

xvals = arange(-10,10,0.001)
f1vals = [f1(x) for x in xvals]
g1vals = [g1(x,0.1) for x in xvals]
error1 = [fabs(f1(x)-g1(x,0.1)) for x in xvals]
g2vals = [g1(x,0.01) for x in xvals]
error2 = [fabs(f1(x)-g1(x,0.01)) for x in xvals]

plt.plot(xvals, f1vals, label="f'(x)")
plt.plot(xvals, g1vals, label="h = 0.1")
plt.plot(xvals, error1, label="error")
plt.legend()
plt.show()

plt.plot(xvals, f1vals, label="f'(x)")
plt.plot(xvals, g2vals, label="h = 0.01")
plt.plot(xvals, error2, label="error")
plt.legend()
plt.show()

# question 7

def g2(x,h):
    return (f(x+h)-f(x))/h

def j1(x,h):
    return fabs(f1(x) - g2(x,h))

def f2(x): # 2nd derivative
    return -18*x*sin(3*x)+12*cos(3*x)+6

def k1(x,h):
    return f2(x)-(f1(x+h)-f1(x))/h

h = array([0.1, 0.01, 0.001, 0.000001])
maxes = array([])

for i in h:
    def k2(x):
        return k1(x,i)

    zeros = findall(k2,-10,10,10000)
    zeros.append(float(-10))
    zeros.append(float(10))

    checked = array([])

    for j in zeros:
        checked = append(checked, j1(j,i))
    
    maxes = append(maxes, max(checked))

print("The answer to question 7 is:" , maxes)

# question 8

def j2(x,h):
    return fabs(f1(x) - g1(x,h))

def k3(x,h):
    return f2(x)-(f1(x+h)-f1(x-h))/(2*h)

h = array([0.1, 0.01, 0.001, 0.0001])
maxes = array([])

for i in h:
    def k4(x):
        return k3(x,i)

    zeros = findall(k4,-10,10,10000)
    zeros.append(float(-10))
    zeros.append(float(10))

    checked = array([])

    for j in zeros:
        checked = append(checked, j2(j,i))

    maxes = append(maxes, max(checked))

print("The answer to question 8 is:" , maxes)

# question 9

def f3(x): # 3rd derivative
    return -54*sin(3*x)-54*x*cos(3*x)

def g3(x,h):
    return (f(x+h)+f(x-h)-2*f(x))/(h**2)

def j3(x,h):
    return fabs(f2(x) - g3(x,h))

def k5(x,h):
    return f3(x)-((f1(x+h)+f1(x-h)-2*f1(x))/(h**2))

h = array([0.1, 0.01, 0.001, 0.0001])
maxes = array([])

for i in h:
    def k6(x):
        return k5(x,i)

    zeros = findall(k6,-10,10,10000)
    zeros.append(float(-10))
    zeros.append(float(10))

    checked = array([])

    for j in zeros:
        checked = append(checked, j3(j,i))

    maxes = append(maxes, max(checked))

print("The answer to question 9 is:" , maxes)