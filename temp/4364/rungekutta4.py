def rk4(f,h,n,t0,U0):
    U = U0
    t = t0
    for i in range(n):
        k1 = h*f(t,U)
        k2 = h*f(t + h/2,U + k1/2)
        k3 = h*f(t + h/2,U + k2/2)
        k4 = h*f(t + h,U + k3)
        U = U + (k1 + 2*k2 + 2*k3 + k4)/6
        t = t + h
    return U

def rk4sys(F,h,n,t0,U0):
    U = U0
    t = t0
    k = len(U)
    for i in range(n):
        k1 = [h*F(t,U)[j] for j in range(k)]
        Z = [U[j] +k1[j]/2 for j in range(k)]
        k2 = [h*F(t+h/2,Z)[j] for j in range(k)]
        Z = [U[j] + k2[j]/2 for j in range(k)]
        k3 = [h*F(t+h/2,Z)[j] for j in range(k)]
        Z = [U[j]+k3[j] for j in range(k)]
        k4 = [h*F(t+h,Z)[j] for j in range(k)]
        U = [U[j]+(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6 for j in range(k)]
        t = t + h
    return U