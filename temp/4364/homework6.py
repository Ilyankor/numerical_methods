from numpy import array, sin, cos, empty, random, linalg, vstack, all, zeros, transpose, matmul
from gradientdescent import newton, graddes

# question 4

def F(U):
    x = U[0]
    y = U[1]
    return array([
        2*x + 3*y*sin(x + y) + 0.1*cos(x) - 1,
        y - sin(x - y) + (2*x**2 + y**2)**(-1) - 1.5
    ])

def dF(U):
    x = U[0]
    y = U[1]
    return array([
        [2 + 3*y*cos(x + y) - 0.1*sin(x),           3*sin(x + y) + 3*y*cos(x + y)],
        [-cos(x - y) - 4*x*(2*x**2 + y**2)**(-2),   1 + cos(x - y) - 2*y*(2*x**2 + y**2)**(-2)]
    ])

print("The answer to question 4 is:", newton(F,dF,[0.5,0],10**(-10)))

# question 5

init_guess_5 = array([
    [-0.2, -0.6],
    [0.2, -0.5],
    [0.5, 0],
    [2, 1.7],
    [3.2, 2.2]
])

solutions_5 = empty((5,2))
for i in range(5):
    solutions_5[i] = newton(F,dF,init_guess_5[i,:],10**(-10))

print("The answer to question 5 is:", solutions_5)

# question 6

def G(U):
    x = U[0]
    y = U[1]
    return array([
        sin(x**2 + y**2) + x*y - 2*x + 2*y,
        x**4 - 8.8*x*y + y**4 + 0.5
    ])

def dG(U):
    x = U[0]
    y = U[1]
    return array([
        [2*x*cos(x**2 + y**2) + y - 2,      2*y*cos(x**2 + y**2) + x + 2],
        [4*x**3 - 8.8*y,                    - 8.8*x + 4*y**3]
    ])

init_guess_6 = array([
    [-1.2, -2.1],
    [-0.2, -0.3],
    [0.3, 0.2],
    [2.1, 1.2]
])

solutions_6 = empty((4,2))
for i in range(4):
    solutions_6[i] = newton(G,dG,init_guess_6[i,:],10**(-10))

print("The answer to question 6 is:", solutions_6)

# question 7

def J(U):
    x = U[0]
    y = U[1]
    z = U[2]
    return -1/3*x**2 - y**2 - 1/2*z**2 + cos(2*x + 2*y - 3*z + 1)

def gradJ(U):
    '''the gradient of the function'''
    x = U[0]
    y = U[1]
    z = U[2]
    return array([
        -2/3*x - 2*sin(2*x + 2*y - 3*z + 1),
        -2*y - 2*sin(2*x + 2*y - 3*z + 1),
        -z + 3*sin(2*x + 2*y - 3*z + 1)
    ])

def d2J(U):
    '''the second derivative of the function'''
    x = U[0]
    y = U[1]
    z = U[2]
    return array([
        [-2/3 - 4*cos(2*x + 2*y - 3*z + 1),     - 4*cos(2*x + 2*y - 3*z + 1),       6*cos(2*x + 2*y - 3*z + 1)],
        [- 4*cos(2*x + 2*y - 3*z + 1),          -2 - 4*cos(2*x + 2*y - 3*z + 1),    6*cos(2*x + 2*y - 3*z + 1)],
        [6*cos(2*x + 2*y - 3*z + 1),            6*cos(2*x + 2*y - 3*z + 1),         -1 - 9*cos(2*x + 2*y - 3*z + 1)]
    ])

def search_7(n,f,df):
    ulist = []
    for i in range(n):
        u = array([2*random.rand(),2*random.rand(),2*random.rand()])
        u = newton(f,df,u,10**(-14))
        if linalg.norm(f(u),2)<10**(-7):
            ulist.append(u)
    return ulist

ulist_7 = search_7(1000,gradJ,d2J)

def remove_duplicates(ulist,f,df):
    vlist = []
    for i in range(len(ulist)):
        vlist.append([round(v,5) for v in ulist[i]])
    cpset = set(tuple(v) for v in vlist)
    cplist = list(list(v) for v in cpset)
    finalcplist = [newton(f,df,v,10**(-14)) for v in cplist]
    return vstack(finalcplist)

points_7 = remove_duplicates(ulist_7,gradJ,d2J)

for i in range(points_7.shape[0]):
    x = points_7[i,0]
    y = points_7[i,1]
    z = points_7[i,2]
    if x**2 + y**2 + z**2 > 4:
        points_7[i,:] = 0
points_7 = points_7[~all(points_7 == 0, axis=1)]

print('''The answer to quesion 7 is:
The critical points are located at: \n''',
points_7)

print("The determinants of the principal minors are:")
for i in points_7:
    print([linalg.det(d2J(i)[0:j,0:j]) for j in range(1,4)])

print("Evaluating the function gives:")
function_value7 = [J(i) for i in points_7]
print(function_value7)

print("The maximum value of f is: \n",
max(function_value7))

# question 8

def K(U):
    return U[0]**2 - 2*U[0] + 3*U[1] + 3*U[1]**2 - U[0]*U[1] - 1

def dK(U):
    return array([
        2*U[0] - 2 - U[1],
        3 + 6*U[1] - U[0]
    ])

print("The answer to question 8 is:", K(graddes(K,dK,[0,2])))

# question 9

def L(U):
    x = U[0]
    y = U[1]
    return 2*x**2 - 1.05*x**4 + 1/6*x**6 + x*y + y**2

def gradL(U):
    x = U[0]
    y = U[1]
    return array([
        4*x - 4*1.05*x**3 + x + y,
        x + 2*y
    ])

def d2L(U):
    x = U[0]
    y = U[1]
    return array([
        [4 - 12*1.05*x**2 + 1,  1],
        [1,                     2]
    ])

def search_9(n,f,df):
    ulist = []
    for i in range(n):
        u = array([10*random.rand()-5,10*random.rand()-5])
        u = graddes(f,df,u)
        if linalg.norm(df(u),2)<10**(-7):
            ulist.append(u)
        ulist.append(u)
    return ulist

ulist_9 = search_9(500,L,gradL)

ulist_9_2 = remove_duplicates(ulist_9,gradL,d2L)

ulist_9_3 = remove_duplicates(ulist_9_2,gradL,d2L)

function_value9 = [L(ulist_9_3[i]) for i in range(3)]

print('''The answer to question 9 is:,
The critical points are located at:''',
ulist_9_3,
"Evaluating the functions gives \n",
function_value9,
"The minimum value of f is: \n",
min(function_value9))

# question 10

def M(U):
    x = U[0]
    y = U[1]
    return 1/3*x**6 - 2.1*x**4 + 4*x**2 + x*y + 4*y**4 - 4*y**2

def gradM(U):
    x = U[0]
    y = U[1]
    return array([
        2*x**5 - 4*2.1*x**3 + 8*x + y,
        x + 16*y**3 - 8*y
    ])

def d2M(U):
    x = U[0]
    y = U[1]
    return array([
        [10*x**4 - 12*2.1*x**2 + 8,     1],
        [1,                             48*y**2 - 8]
    ])

def search_10(n,f,df):
    ulist = []
    for i in range(n):
        u = array([6*random.rand()-3,4*random.rand()-2])
        u = graddes(f,df,u)
        if linalg.norm(df(u),2)<10**(-7):
            ulist.append(u)
    return ulist

ulist_10 = search_10(1000,M,gradM)

ulist_10_2 = remove_duplicates(ulist_10,gradM,d2M)

function_value10 = [M(ulist_10_2[i]) for i in range(6)]

print("The answer to question 10 is: \n",
"The critical points are located at: \n",
ulist_10_2,
"Evaluating the function at the critical points give:",
function_value10,
"The minimum of f is",
min(function_value10))



# question 12

A_12 = zeros((10,10))

for i in range(10):
    A_12[i,i] = -2

for i in range(9):
    A_12[i,i+1] = 1
    A_12[i+1,i] = 1

print('''The answer to question 12 is:
It is negative semidefinite because the determinants of the principal minors of A are: \n''',
[linalg.det(A_12[0:i,0:i]) for i in range(1,11)])

# question 13

C = array([
    [5, -1, 1, 2],
    [-1, 6, 2, 1],
    [1, 2, 5, -1],
    [2, 1, -1, 7]
])

v = transpose(array([
    1,3,-4,2
]))

def P(U):
    x = array(U)
    return 0.5*matmul(transpose(x),C,x) - matmul(transpose(v),x) - 7

invC = linalg.inv(C)

print("The answer to question 13 is: \n",
"The critical point is located",
matmul(invC,v),
"The absolute minimum is: -12.010600706713781")