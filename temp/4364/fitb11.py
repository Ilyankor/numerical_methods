from rungekutta4 import rk4sys
from bisection import bisect
from numpy import sin, cos, random, array, empty, linalg, vstack, sum
from gradientdescent import graddes, newton

# question 1

def F(t,U):
    return [0.13*U[1]*U[2] - 0.25*U[0], 
            0.13*U[1]*U[2] - 0.25*U[1],
            0.63 + 0.25*U[0] - 0.26*U[1]*U[2]
            ]

print("The answer to question 1 is:", rk4sys(F,0.01,500,0,[1.32,2.61,14.16])[2])

# question 2

'''
The determinants of the principal minors are:
4 + y
y^2 + 9y + 19
y^3 + 10y^2 + 23y - 9
y^4 + 11y^3 + 19y^2 - 65y - 10
'''

def g(y):
    return y**4 + 11*y**3 + 19*y**2 - 65*y - 10

print("The answer to question 2 is:",bisect(g,0,2)) 

# question 3 and 4

'''
The gradient of the function is:
-sin(x)cos(y) - 2/15*x + 1
-sin(y)cos(x) - 2/15*y

The second derivative of the function is:
-cos(x)cos(y) - 2/15        sin(x)sin(y)
sin(x)sin(y)               -cos(x)cos(y) - 2/15

We want grad(f) = 0. The intersections lie in [0,12] x [-6,6].

Generate random points, use Newton's method to find the critical points.
Evaluate the critical points to see which are local minima, maxima
'''

def P(U):
    return array([
        -sin(U[0])*cos(U[1]) - 2/15*U[0] + 1,
        -sin(U[1])*cos(U[0]) - 2/15*U[1]
    ])

def dP(U):
    return array([
        [-cos(U[0])*cos(U[1]) - 2/15,   sin(U[0])*sin(U[1])],
        [sin(U[0])*sin(U[1]),           -cos(U[0])*cos(U[1]) - 2/15]
    ])

"""
randx = 12*random.rand(1000)
randy = 12*random.rand(1000) - 6

newtonresult = empty((1000, 2))

for i in range(1000):
    newtonresult[i] = newton(P,dP,[randx[i],randy[i]],10**(-14))

gradient_eval = empty((1000,2))

for i in range(1000):
    gradient_eval[i] = P(newtonresult[i])
"""

def hunt(n):
    ulist = []
    for i in range(n):
        u = array([12*random.rand(),12*random.rand()-6])
        u = newton(P,dP,u,10**(-14))
        if linalg.norm(P(u),2)<10**(-7):
            ulist.append(u)
    return ulist

ulist = hunt(1000)
vlist = []

for i in range(len(ulist)):
    vlist.append([round(v,5) for v in ulist[i]])

cpset = set(tuple(v) for v in vlist)
cplist = list(list(v) for v in cpset)
finalcplist = [newton(P,dP,v,10**(-14)) for v in cplist]
final_array = vstack(finalcplist)

def second_derivative_test(array):
    mins = []
    maxes = []
    for i in range(len(finalcplist)):
        u = array[i,:]
        entry_1_1 = dP(u)[1,1]
        determinant = linalg.det(dP(u))
        if determinant > 0:
            if entry_1_1 > 0:
                mins.append(u)
            else: maxes.append(u)
    return mins, maxes
mins, maxes = second_derivative_test(final_array)

print("The answer to question 3 is:", sum(vstack(mins),axis=0)[0]/len(mins))

print("The answer to question 4 is:", sum(vstack(maxes),axis=0)[0]/len(maxes))

# question 5

print("The answer to question 5 is:", sum(final_array,axis=0)[0]/len(finalcplist))

# question 6

'''The maximum of the function happens at y=0.

def j(x):
    return -sin(x) - 2/15 * x + 1

def m(x,y):
    return cos(x)*cos(y) - 1/15 *(x**2 + y**2) + x -1

xval = bisect(j,6,7)
print("The answer to question 6 is:", m(xval,0))'''

def OG(U):
    return cos(U[0])*cos(U[1]) - 1/15 * (U[0]**2 + U[1]**2) + U[0] - 1

values = []
for i in range(len(finalcplist)):
    values.append(OG(finalcplist[i]))

print("The answer to question 6 is:", max(values))

# question 7

def Q(U):
    return U[0]**2 - 2*U[0] + 3*U[1] + 3*U[1]**2 - U[0]*U[1] - 1

def dQ(U):
    return array([
        2*U[0] - 2 - U[1],
        3 + 6*U[1] - U[0]
    ])

print("The answer to question 7 is:", graddes(Q,dQ,[0,2])[0])
print("The answer to question 8 is:", graddes(Q,dQ,[0,2])[1])


