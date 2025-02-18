import numpy as np


# pdf and cdf of an image
def hist_info(src:np.ndarray) -> np.ndarray:
    px = src.shape[0] * src.shape[1]
    pdf = np.reshape(cv.calcHist([src], [0], None, [256], [0,256]) / px, 256)
    cdf = np.cumsum(pdf)
    return [pdf, cdf]

# question 3b

img0 = cv.imread("brain_image.png", cv.IMREAD_GRAYSCALE)
cv.imwrite('partb_img.png', img0)

[img0_pdf, img0_cdf] = hist_info(img0)
intensity = range(256)

# graph pdf and cdf
plt.plot(intensity, img0_pdf, linewidth=1.5)
plt.title("PDF", fontsize=16)
plt.xlabel("intensity", fontsize=14)
plt.savefig("partb_pdf.pdf")
plt.close()

plt.plot(intensity, img0_cdf, linewidth=1.5)
plt.title("CDF", fontsize=16)
plt.xlabel("intensity", fontsize=14)
plt.savefig("partb_cdf.pdf")
plt.close()

# question 3c

img0_eq = cv.equalizeHist(img0)
cv.imwrite('partc_img.png', img0_eq)

img0_eq_cdf = hist_info(img0_eq)[1]

# graph cdf
plt.plot(intensity, img0_eq_cdf, linewidth=1.5)
plt.title("Equalized CDF", fontsize=16)
plt.xlabel("intensity", fontsize=14)
plt.savefig("partc_cdf.pdf")
plt.close()

# question 3e

img1 = cv.imread("brainT1.png", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("brainT2.png", cv.IMREAD_GRAYSCALE)

img1_matched = skimage.exposure.match_histograms(img1, img2)
img1_matched = img1_matched.astype(np.uint8)

img1_cdf = hist_info(img1)[1]
img2_cdf = hist_info(img2)[1]
img1_matched_cdf = hist_info(img1_matched)[1]

cv.imwrite('parte_img.png', img1_matched)
y = np.transpose(np.array([img1_cdf, img2_cdf, img1_matched_cdf]))

# graph cdfs
plt.plot(intensity, y, linewidth=1.5)
plt.title("Matched histogram CDFs", fontsize=16)
plt.xlabel("intensity", fontsize=14)
plt.legend(["T1", "T2", "matched"])
plt.savefig("parte_cdf.pdf")
plt.close()


def butterworth(src:np.ndarray, w:float, n:int) -> np.ndarray:
    c1, c2 = img_center(src)
    x, y = np.meshgrid(range(src.shape[0]), range(src.shape[1]), 
        indexing='ij')
    H = np.power(1 + np.power(((x - c1)**2 + (y - c2)**2)/(w**2), n), -1)
    return H


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