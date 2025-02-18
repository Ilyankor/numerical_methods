import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage

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