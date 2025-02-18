# question 1a

from numpy import array, append

p = array([2])

for i in range(6):
    p = append(p, p[i]/2 + 1/p[i])

print(p)

# question 1b

from numpy import array, append

p = array([3])

for i in range(6):
    x = p[i]/2 + 3/(2*p[i])
    p = append(p,x)

print(p)

# question 3

from bisection import findall # automated bisection script
from numpy import sin, cos, array, append

def g(x):
    return a*x**2+0.5*cos(5*x)+20*sin(x)

p = array([])

for i in range(1,11):
    a = 0.001*i
    p = append(p,len(findall(g,-1000,1000,10000)))

print(p)
