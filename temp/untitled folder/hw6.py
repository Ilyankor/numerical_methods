import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

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

I = plt.imread("cells.jpg")

# thresh, I_val, I_sb = otsu(I)

# plt.rcParams["text.usetex"] = True
# plt.plot(I_val, I_sb)
# plt.xlabel("intensity", fontsize=14)
# plt.ylabel(r'$\sigma_B^2$', fontsize=14)
# plt.savefig("img3b.pdf")

# I_seg = np.where(I <= thresh, 0, 1)
# plt.imshow(I_seg, cmap="gray")
# plt.savefig("img3c.pdf")
# plt.show()

# n = 0
# for i in range(1,8):
#     for j in range(1,11):
#         I_seg = ski.segmentation.chan_vese(I, mu=(0.015 + 0.001*i), lambda1=(1.0 + 0.03*j), lambda2=(1.0 + 0.03*j))
#         plt.imshow(I_seg, cmap="gray")
#         plt.savefig(f"testo/I_{str(i)}_{str(j)}.pdf")
#         n += 1
#         print(n)

I_seg = ski.segmentation.chan_vese(I, mu=0.017, lambda1=1.06, lambda2=1.06)
plt.imshow(I_seg, cmap="gray")
plt.savefig("img3d.pdf")