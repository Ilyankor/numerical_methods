import numpy as np


# gradient
def gradient(src:np.ndarray):
    dx_filter = np.transpose(np.array([[0, -1, 1]]))
    dy_filter = np.array([[0, -1, 1]])
    dx = sci.ndimage.correlate(src, dx_filter)
    dy = sci.ndimage.correlate(src, dy_filter)
    norm = np.sqrt(dx**2 + dy**2)
    return dx, dy, norm

# weight function
def weight(src:np.ndarray) -> np.ndarray:
    b = np.ones_like(src)
    grad_I = gradient(src)[2]
    b[1:-1, 1:-1] = (1 / np.sqrt(np.ones_like(grad_I) + grad_I))[1:-1, 1:-1]
    return b

# energy function
def energy(x:np.ndarray, var) -> float:
    b, L, I0 = var
    return np.sum(b * gradient(x)[2]**2) + L * np.sum((x - I0)**2)

# Euler-Lagrange equation
def Euler_Lagrange(x:np.ndarray, var):
    b, L, I0 = var
    return -1.0*b*sci.ndimage.laplace(x) - (gradient(b)[0]*gradient(x)[0] 
        + gradient(b)[1]*gradient(x)[1]) + L*x

# gradient descent
def grad_descent(df, var, rhs, x0:np.ndarray, eps:float, tol:float, E):
    x = x0
    r = rhs - df(x, var)
    e = np.array([E(x, var)]) # initialize energy
    i = 0 # initialize count
    for j in range(10**6): # maximum iterations
        x = x + eps * r
        r = rhs - df(x, var)
        en = E(x, var)
        if (np.abs(en - e[-1])/en) < tol: # stopping criteria
            e = np.append(e, en)
            i+=1
            break
        e = np.append(e, en)
        i+=1
    return x, i, e
           
# conjugate gradient descent
def conj_descent(df, var, rhs, x0:np.ndarray, tol:float, E):
    N = np.size(x0) # vector size

    x = x0
    r = rhs - df(x, var)
    p = r
    rtr = np.dot(np.reshape(r, N), np.reshape(r, N)) # r transpose r
    e = np.array([E(x, var)]) # initialize energy
    i = 0 # initialize count
    for j in range(N):
        alpha = rtr / np.dot(np.reshape(p, N), df(p, var).reshape(N))
        x = x + alpha * p
        r = rhs - df(x, var)
        en = E(x, var)
        if (np.abs(en - e[-1])/en) < tol:
            e = np.append(e, en)
            i += 1
            break
        else:
            rtr_new = np.dot(np.reshape(r, N), np.reshape(r, N))
            beta = rtr_new / rtr
            p = r + beta * p
            rtr = rtr_new
            e = np.append(e, en)
            i += 1
    return x, i, e



# Otsu's method
def otsu(src:np.ndarray):
    n = np.size(src)
    temp = src.reshape(n)

    img_min = temp.min()
    img_max = temp.max()
    img_val = range(img_min, img_max+1)
    img_hist = np.histogram(temp, range(img_min, img_max+2))[0]/n

    mg = np.sum(img_val * img_hist)

    sb = []
    for i in range(1, img_max + 2 - img_min):
        c1 = img_hist[0:i]
        c2 = img_hist[i:]

        p1 = np.sum(c1)
        p2 = np.sum(c2)

        m1 = np.sum(c1 * img_val[0:i])/(p1)
        if p2 == 0:
            m2 = 0
        else:
            m2 = np.sum(c2 * img_val[i:])/(p2) 
        
        sb.append(p1*(m1-mg)**2 + p2*(m2-mg)**2)
    return img_val[np.argmax(sb)], img_val, sb