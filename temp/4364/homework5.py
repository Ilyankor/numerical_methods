from firstorderdiffeq import euler, midpoint, modeuler
from taylor import taylor3
from rungekutta4 import rk4, rk4sys
from bisection import bisect
from numpy import sin, cos, exp, absolute, arange, empty, savetxt, linspace
import matplotlib.pyplot as plt

# question 1

def solution(t):
    return 1/101*(-1010 + 1121*exp(t/10) - 10*cos(t) + 100*sin(t))

def f(t,y):
    return 1 + cos(t) + 0.1*y

def ft(t,y):
    return -sin(t)

def fy(t,y):
    return 0.1

def fty(t,y): # = fyy
    return 0

def ftt(t,y):
    return -cos(t)

table1 = empty([6,6])

for i in range(6):
    t = 10*i
    n = 100*i
    table1[i,0] = solution(t)
    table1[i,1] = euler(f,0.1,n,0,1)
    table1[i,2] = midpoint(f,0.1,n,0,1)
    table1[i,3] = modeuler(f,0.1,n,0,1)
    table1[i,4] = taylor3(f,ft,fy,fty,fty,ftt,0.1,n,0,1)
    table1[i,5] = rk4(f,0.1,n,0,1)

savetxt("table1.csv", table1, delimiter=',')

table2 = empty([6,5])

for i in range(6):
    t = 10*i
    n = 100*i
    table2[i,0] = absolute(table1[i,0] - table1[i,1])
    table2[i,1] = absolute(table1[i,0] - table1[i,2])
    table2[i,2] = absolute(table1[i,0] - table1[i,3])
    table2[i,3] = absolute(table1[i,0] - table1[i,4])
    table2[i,4] = absolute(table1[i,0] - table1[i,5])

savetxt("table2.csv", table2, delimiter=',')


# question 2

def F(x,U):
    return [U[1],U[0]**2-40*U[0]]

def G(a):
    return rk4sys(F,0.01,100,0,[a,0])[0]

xvals = arange(-20,40,0.005)
fvals = [G(x) for x in xvals]
plt.plot([-20,40],[0,0])
plt.plot(xvals,fvals,label='f(x)')
plt.savefig("figure1.svg", format="svg")

plt.clf()

xvals1 = arange(0,101)
xvals2 = arange(0,1.01,0.01)
fvals1 = [rk4sys(F,0.01,x,0,[bisect(G,-18,-16),0])[0] for x in xvals1]
fvals2 = [rk4sys(F,0.01,x,0,[bisect(G,-1,1),0])[0] for x in xvals1]
fvals3 = [rk4sys(F,0.01,x,0,[bisect(G,35,37),0])[0] for x in xvals1]
fvals4 = [rk4sys(F,0.01,x,0,[bisect(G,38,40),0])[0] for x in xvals1]
plt.plot(xvals2,fvals1,label='f1(x)')
plt.plot(xvals2,fvals2,label='f2(x)')
plt.plot(xvals2,fvals3,label='f3(x)')
plt.plot(xvals2,fvals4,label='f4(x)')
plt.savefig("figure2.svg", format="svg")

plt.clf()

# question 3

def taylor4(f,ft,fy,fty,fyy,ftt,ftty,ftyy,fttt,fyyy,h,n,t0,y0):
    y = y0
    t = t0
    for i in range(n):
        z1 = ft(t,y) + fy(t,y)*f(t,y)
        z2 = ftt(t,y) + 2*fty(t,y)*f(t,y) + z1*fy(t,y) + fyy(t,y)*f(t,y)**2
        z3 = fttt(t,y) + 3*ftty(t,y)*f(t,y) + 3*ftyy(t,y)*f(t,y)**2 + 3*fty(t,y)*z1 + 3*fyy(t,y)*f(t,y)*z1 + fyyy(t,y)*f(t,y)**3
        y = y + f(t,y)*h + 0.5*z1*h**2 + z2*h**3/6.0 + (z3 + z2*fy(t,y))*h**4/24
        t = t + h
    return y

def fttt(t,y):
    return sin(t)

table3 = empty([6,3])

for i in range(6):
    t = 10*i
    n = 100*i
    table3[i,0] = solution(t)
    table3[i,1] = taylor4(f,ft,fy,fty,fty,ftt,fty,fty,fttt,fty,0.1,n,0,1)
    table3[i,2] = rk4(f,0.1,n,0,1)

savetxt("table3.csv", table3, delimiter=',')

# question 4

def F1(t,U):
    return [U[1],-1 + 0.016*U[1]**2 - 0.112*U[1]]

def F2(t,U):
    return [U[1],1 - 0.016*U[1]**2 - 0.112*U[1]]

# rough estimate

def func1(t):
    n1 = int(t*100)
    v1 = rk4sys(F1,0.01,n1,0,[40,4])[1]
    x1 = rk4sys(F1,0.01,n1,0,[40,4])[0]
    n2 = 0
    while rk4sys(F2,0.01,n2,t,[x1,v1])[1] < 0:
        n2 = n2 + 10
    print(n2)
    return rk4sys(F2,0.01,n2,t,[x1,v1])[0]-20

# t1 approx. 10.89, t2 approx. 3.72
# refine the estimate by decreasing the step size to 10^-5 and bisection

def func2(t):
    n1 = int(t*100000)
    v1 = rk4sys(F1,0.00001,n1,0,[40,4])[1]
    x1 = rk4sys(F1,0.00001,n1,0,[40,4])[0]
    n2 = 372450
    while rk4sys(F2,0.00001,n2,t,[x1,v1])[1] < 0:
        n2 = n2 + 1
    print([n2,t]) # for progress checking
    return rk4sys(F2,0.00001,n2,t,[x1,v1])[0]-20

# print(bisect(func2,10.884,10.885)) # commented out to save time

# refined estimate: t1 = 10.88464, t2 = 3.72456

t1 = 10.88464
t2 = 3.72456

def s1(t):
    n = int(t*100000)
    return rk4sys(F1,0.00001,n,0,[40,4])

x1 = s1(t1)[0]
v1 = s1(t1)[1]

def s2(t):
    n = int(t*100000)
    return rk4sys(F2,0.00001,n,t1,[x1,v1])

#tvals1 = linspace(0,t1,500)
#tvals2 = linspace(0,t2,171)
#fvals1 = [s1(t)[0] for t in tvals1]
#fvals2 = [s2(t)[0] for t in tvals2]
#plt.plot(tvals1,fvals1,label='f1(x)',color="blue")
#plt.plot(tvals2+t1,fvals2,label='f2(x)',color="blue")
#plt.savefig("figure3.svg", format="svg")
#plt.show()

tvals1 = linspace(0,t1,500)
tvals2 = linspace(0,t2,171)
fvals1 = [s1(t)[1] for t in tvals1]
fvals2 = [s2(t)[1] for t in tvals2]
plt.plot(tvals1,fvals1,label='f1(x)',color="blue")
plt.plot(tvals2+t1,fvals2,label='f2(x)',color="blue")
#plt.savefig("figure3.svg", format="svg")
plt.show()