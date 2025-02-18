import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import skimage as ski
import scipy.io as io

# question 1 bonus

# function
def g(x:float, n:int) -> float:
    val = 0.5 + 1/np.pi \
        * integrate.quad(lambda t: np.sinc(t/np.pi), 0, np.pi*n*x)[0]
    return val

g_n = np.vectorize(g)

# graphing
x = np.linspace(-2, 2, 401)

plt.rcParams['text.usetex'] = True
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True)
fig1.suptitle(r'Graphs of $g_n(x)$', fontsize=16)
ax1.plot(x, g_n(x, 1))
ax1.set_title(r'$n=1$')
ax2.plot(x, g_n(x, 5))
ax2.set_title(r'$n=5$')
ax3.plot(x, g_n(x, 10))
ax3.set_title(r'$n=10$')
ax4.axis('off')

plt.savefig('part_1.pdf')
plt.clf
plt.close()


# question 2

# part a
img_1 = ski.io.imread("test_image_coins.png")
img_1_ft = np.fft.fftshift(np.fft.fft2(img_1))
img_1_log = np.log(1 + np.abs(img_1_ft))

plt.cla()
plt.imshow(img_1_log, cmap='gray')
plt.savefig('part_2a.pdf')

# part b
def img_center(src:np.ndarray) -> tuple:
    c1 = np.ceil(0.5*src.shape[0])
    c2 = np.ceil(0.5*src.shape[1])
    return c1, c2

def keep_radius(src:np.ndarray, r:int) -> np.ndarray:
    c1, c2 = img_center(src)

    new = np.zeros((src.shape[0],src.shape[1]), dtype=np.cdouble)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if np.abs(c1 - i) <= r and np.abs(c2 - j) <= r:
                new[i,j] = src[i,j]
            else:
                continue
    return new

img_1_comp_ft = keep_radius(img_1_ft, 50)
img_1_comp = np.real(np.fft.ifft2(np.fft.ifftshift(img_1_comp_ft)))

plt.cla()
plt.imshow(img_1_comp, cmap='gray')
plt.savefig('part_2b.pdf')

# part d
def butterworth(src:np.ndarray, w:float, n:int) -> np.ndarray:
    c1, c2 = img_center(src)
    x, y = np.meshgrid(range(src.shape[0]), range(src.shape[1]), 
        indexing='ij')
    H = np.power(1 + np.power(((x - c1)**2 + (y - c2)**2)/(w**2), n), -1)
    return H

# part e
img_1_ft2 = np.multiply(img_1_ft, butterworth(img_1, 40, 3))
img_1_comp_ft2 = keep_radius(img_1_ft2, 50)
img_1_comp2 = np.real(np.fft.ifft2(np.fft.ifftshift(img_1_comp_ft2)))

plt.cla()
plt.imshow(img_1_comp2, cmap='gray')
plt.savefig('part_2e.pdf')


# question 3
img_2 = io.loadmat("mysterious_Radon.mat")['R']
img_2_rt = ski.transform.iradon(img_2)

fig2, ((ax1, ax2)) = plt.subplots(1, 2, constrained_layout=True)
ax1.set_title(r'$\mathcal{R}I$', fontsize=16)
ax2.set_title(r'$I$', fontsize=16)
ax1.imshow(img_2, cmap='gray')
ax2.imshow(img_2_rt, cmap='gray')

plt.savefig('part_3.pdf')
plt.close