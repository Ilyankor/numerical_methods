import numpy as np
import scipy as sci
import skimage as ski
import matplotlib.pyplot as plt
import itertools

# create a small patch of radius r centered at px
def small_patch(src:np.ndarray, px:tuple, r:int):
    left_1, left_2 = [max([0, x - r]) for x in px]
    right_1, right_2 = [min([px[i] + r + 1, src.shape[i]]) for i in range(2)]
    return src[left_1:right_1, left_2:right_2]

# slice an image horizontally
def horz_slice(src:np.ndarray):
    center = int(np.floor(0.5*src.shape[0]))
    return src[center, :]

# estimate length of blur over vertical edges
def estimate_blur(src:np.ndarray, min_dx:float):
    length = []
    for i in range(src.shape[0]):
        dx = np.abs(sci.ndimage.correlate(src[i, :], 0.5*np.array([-1, 0, 1]),
            mode='nearest'))
        condition = np.where(dx >= min_dx, True, False)
        temp = []
        for x, group in itertools.groupby(condition):
            if True:
                temp.append(sum(group))
        length.append(max(temp))
    return np.average(length)

# ideal image
def ideal_edge(x:float, y:float):
    if x < 0:
        return -1.0
    else:
        return 1.0

# 2d gaussian
def gaussian_2d(x:float, y:float, s:float):
    return (1.0 / (2*np.pi*s**2)) * np.exp(-0.5*(x**2 + y**2)/(s**2))

# blurred ideal edge image
def gaussian_convolved(x0:np.ndarray, y0:np.ndarray, s:float):
    val = np.zeros_like(x0)
    for i in range(len(x0)):
        f = lambda x, y: gaussian_2d(x, y, s) * ideal_edge(x0[i]-x, y0[i]-y)
        val[i] = sci.integrate.dblquad(f, -np.inf, np.inf, -np.inf, np.inf)[0]
    return val

# gaussian_filter
def gaussian_filter(size:tuple, std:float):
    m, n = [int(np.floor(0.5*(x - 1))) for x in size]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x**2 + y**2)/(2*std**2))
    h /= h.sum()
    return h

# HJ
def discrete_conv(src:np.ndarray, psf:np.ndarray):
    p = int((np.shape(psf)[0] - 1)/2)
    res = sci.ndimage.correlate(src, psf,
        mode="constant", cval = 1.0)[p:-p, p:-p]
    return res

# H*J
def discrete_conv_star(src: np.ndarray, psf:np.ndarray):
    p = int((np.shape(psf)[0] - 1)/2)
    src = np.pad(src, 2*p, 'constant', constant_values=(1, 1))
    res = discrete_conv(src, psf)
    return res

# Lucy-Richardson algorithm
def lucy_richardson(src:np.ndarray, psf:np.ndarray, num_iter:int):
    p = int((np.shape(psf)[0] - 1)/2)
    img_old = np.pad(src, p, 'constant', constant_values=(1, 1))
    for i in range(num_iter):
        img_new = img_old * discrete_conv_star(src
            / (discrete_conv(img_old, psf) + 1e-12), psf)
        img_old = img_new
    return img_new[p:-p, p:-p]

# question 2b
img2b = sci.io.loadmat("blurred1.mat")['I0']
img2b = np.asarray(img2b, dtype="float")
plt.imshow(img2b, cmap="gray")
plt.savefig("img2b_blurry.pdf")
plt.close()

# create slices of radius 15
r = 15
points_to_check_b = [(125, 515), (135, 515), (145, 515), (155, 515), 
    (160, 515), (45, 430), (347, 288), (45, 415)]
slices_b = np.zeros((len(points_to_check_b), 2*r+1))

for i in range(len(points_to_check_b)):
    patch = small_patch(img2b, points_to_check_b[i], r)
    slices_b[i, :] = horz_slice(patch)

# visualize the first slice
plt.plot(slices_b[0, :])
plt.savefig("img2b_slice.pdf")
plt.close()

print("The length of the blur is about",
    (estimate_blur(slices_b, 4.0)))

# question 2c
r = 10
h = np.full((r, r), 1/(r**2))
img2b_01 = ski.exposure.rescale_intensity(img2b, in_range='image',
    out_range=(0, 1))
img2b_dec = ski.restoration.wiener(img2b_01, h, 0.002)
plt.imshow(img2b_dec, cmap="gray")
plt.savefig("img2b_dec.pdf")
plt.close()

# question 2d
m = 6
n = 81

# horizontal slice of radius 40 at y = 0
xn = np.linspace(-40, 40, n)
yn = np.zeros((n))

slices = np.zeros((m, n))
for s in range(m):
    slices[s, :] = gaussian_convolved(xn, yn, 2*s + 2)

# save/load the slice matrix
np.save("mat.npy", slices)
slices = np.load("mat.npy")

# estimate length of blur for varying sigma
length = []
for i in range(m):
    condition = np.where(abs(slices[i, :]) <= 0.999, True, False)
    temp = []
    for x, group in itertools.groupby(condition):
        if True:
            temp.append(sum(group))
    length.append(max(temp))

# visualize the blur
plt.rcParams['text.usetex'] = True
for i in range(m):
    s = 2*i + 2
    plt.plot(xn, slices[i, :], label=r'$\sigma = {}$'.format(s))
plt.xlabel(r'$x$')
plt.legend()
plt.savefig("img2d_sigma.pdf")
plt.close()

print("The length of the blur related to sigma is about",
    np.round(np.average(np.divide(length, 2*np.linspace(1, m, m))),3))

img2d = sci.io.loadmat("blurred2.mat")["I0"]
img2d = np.asarray(img2d, dtype="float")
plt.imshow(img2d, cmap="gray")
plt.savefig("img2d_blurry.pdf")
plt.close()

# create slices of radius 15
r = 15
points_to_check_d = [(120, 470), (50, 330), (140, 145), (130, 145), (60, 330),
    (70, 330), (60, 455), (50, 455), (70, 455), (100, 470), (110, 470),
    (130, 470)]
slices_d = np.zeros((len(points_to_check_d), 2*r+1))

for i in range(len(points_to_check_d)):
    patch = small_patch(img2d, points_to_check_d[i], r)
    slices_d[i, :] = horz_slice(patch)

print("The standard deviation is about",
    np.round(estimate_blur(slices_d, 1.0)/6.576, 3))

# apply Wiener filtering
h = gaussian_filter(np.shape(img2d), 2.902)
img2d_01 = ski.exposure.rescale_intensity(img2d, in_range='image',
    out_range=(0, 1))
img2d_dec = ski.restoration.wiener(img2d_01, h, 0.00508)
plt.imshow(img2d_dec, cmap="gray")
plt.savefig("img2d_dec.pdf")
plt.close()

# question 3d
img3d = ski.io.imread("img3d.png")
img3d = np.asarray(img3d, dtype="float")

for s in [1, 2, 4]:
    h = gaussian_filter((101, 101), s)
    blurred = sci.ndimage.correlate(img3d, h)
    reconstructed = lucy_richardson(blurred, h, 10)

    # visualize an example for various sigma
    plt.rcParams['text.usetex'] = True
    fig1, ((ax1, ax2, ax3)) = plt.subplots(1, 3, constrained_layout=True)
    fig1.suptitle(r'$\sigma = {}$'.format(s), fontsize=16, y=0.8)
    ax1.imshow(img3d, cmap="gray")
    ax1.set_title("Original")
    ax2.imshow(blurred, cmap="gray")
    ax2.set_title("Blurred")
    ax3.imshow(reconstructed, cmap="gray")
    ax3.set_title("Reconstructed")
    plt.savefig(r'img3d_{}.pdf'.format(s))
    plt.close()