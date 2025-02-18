from integration import midpoint, trapezoidal, romberg, simpsons
from numpy import sqrt, tan, zeros

# question 1

def f(x):
    return tan(sqrt(x**2+1))

print("The answer to question 1 is:", midpoint(f,100,0,1.1))

# question 2

print("The answer to question 2 is:", trapezoidal(f,100,0,1.1))

# question 3

def richardsonmidpoint(f,n,a,b,k):
    d = zeros((k,k))
    for i in range(k):
        d[i,0] = midpoint(f,n,a,b)
        for j in range(1,i+1):
            d[i,j] = d[i,j-1] + 1/(4**j-1)*(d[i,j-1] - d[i-1,j-1])
        n = 2*n
    return d[k-1,k-1]

print("The answer to question 3 is:", richardsonmidpoint(f,100,0,1.1,2))

# question 4

def richardsontrapezoidal(f,n,a,b,k):
    d = zeros((k,k))
    for i in range(k):
        d[i,0] = trapezoidal(f,n,a,b)
        for j in range(1,i+1):
            d[i,j] = d[i,j-1] + 1/(4**j-1)*(d[i,j-1] - d[i-1,j-1])
        n = 2*n
    return d[k-1,k-1]

print("The answer to question 4 is:", richardsontrapezoidal(f,100,0,1.1,2))

# question 5

print("The answer to question 5 is:", romberg(f,10**(-6),0,1.1))

# question 7

print("The answer to question 7 is:", simpsons(f,100,0,1.1))