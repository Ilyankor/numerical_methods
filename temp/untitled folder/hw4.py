import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

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

img0 = sci.io.loadmat("test_image_Assignment4.mat")['I0'].astype(float)
b = weight(img0) # weight function
par = (b, 0.1, img0) # parameters

# gradient descent with lambda = 0.1
result_grad, count_grad, E_grad = grad_descent(Euler_Lagrange, par, 
    par[1]*par[2], img0, 0.312, 10**(-4), energy)
print("Gradient descent stopped after", count_grad, "iterations.")

# plot the image
plt.rcParams['text.usetex'] = True
plt.imshow(result_grad, cmap = "gray")
plt.savefig('part_f_1.pdf')
plt.close()

# plot the energy
plt.plot(range(count_grad+1), E_grad, linewidth=2)
plt.title(r'Evolution of energy', fontsize=16)
plt.ylabel(r'energy $E$', fontsize=13)
plt.xlabel(r'iteration $i$', fontsize=13)
plt.savefig('part_f_2.pdf')
plt.close()

# conjugate gradient descent with lambda = 0.1
result_conj, count_conj, E_conj = conj_descent(Euler_Lagrange, par, 
    par[1]*par[2], img0, 10**(-4), energy)
print("Conjugate gradient descent stopped after", count_conj, "iterations.")

# plot the image
plt.imshow(result_conj, cmap = "gray")
plt.savefig('part_g_1.pdf')
plt.close()

# plot the energy
plt.plot(range(count_conj+1), E_conj, linewidth=2)
plt.title(r'Evolution of energy', fontsize=16)
plt.ylabel(r'energy $E$', fontsize=13)
plt.xlabel(r'iteration $i$', fontsize=13)
plt.xticks(range(0,19,3))
plt.savefig('part_g_2.pdf')
plt.close()

# snr vs lambda
imgd = sci.io.loadmat("test_image_Assignment4.mat")['Igd'].astype(float)
imgd_norm = np.linalg.norm(imgd) # norm of groundtruth image
snr = np.zeros(60) # initialize snr
lambdas = np.linspace(0.01, 0.6, 60) # initialize lambdas
for i in range(60):
    par_i = (b, lambdas[i], img0) # set parameters
    img_lambda = conj_descent(Euler_Lagrange, par_i, par_i[1]*par_i[2], 
        img0, 10**(-4), energy)[0]
    snr[i] = imgd_norm / np.linalg.norm(img_lambda - imgd)

print("The highest SNR is", np.max(snr), "at lambda =", 
    lambdas[np.argmax(snr)].round(2))

# plot the snr
plt.plot(lambdas, snr)
plt.title(r'Signal to noise ratio', fontsize=16)
plt.ylabel(r'SNR', fontsize=13)
plt.xlabel(r'$\lambda$', fontsize=13)
plt.savefig('part_h.pdf')
plt.close()