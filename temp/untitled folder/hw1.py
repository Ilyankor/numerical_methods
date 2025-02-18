import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

# sum without pth root
def norm_sum(image: np.ndarray, p: float) -> float:
    m = image.shape[0]
    n = image.shape[1]
    sum_power = (1.0/(m * n)) * np.sum(np.float_power(image, p))
    return sum_power

# L^p norm
def lp_norm(image: np.ndarray, p: float) -> float:
    norm = norm_sum(image, p) ** (1.0/p)
    return norm

# forward difference
def forward_difference(array: np.ndarray) -> np.ndarray:
    m = array.shape[0]
    n = array.shape[1]

    diff = array[1:m, :] - array[0:m-1, :]
    diff = (m - 2) * diff[1:m-1, 1:n-1]
    return diff

# backward difference
def backward_difference(array: np.ndarray) -> np.ndarray:
    m = array.shape[0]
    n = array.shape[1]

    diff = array[1:m, :] - array[0:m-1, :]
    diff = (m - 2) * diff[0:m-2, 1:n-1]
    return diff

# centered difference
def centered_difference(array: np.ndarray) -> np.ndarray:
    m = array.shape[0]
    n = array.shape[1]

    diff = array[2:m, :] - array[0:m-2, :]
    diff = 0.5 * (m - 2) * diff[:, 1:n-1]
    return diff

# norm of gradient vector
def gradient(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    grad = np.sqrt(dx ** 2 + dy ** 2)
    return grad

# W(1,p) norm
def sobolev_norm(image: np.ndarray, p: float, difference) -> float:
    m = image.shape[0]
    n = image.shape[1]

    # replication convention
    padded = np.pad(image, 1, 'edge')

    norm = (norm_sum(image, p) + (1.0/(m * n)) * np.sum(
        gradient(difference(padded), np.transpose(difference(np.transpose(
        padded)))) ** p)) ** (1.0/p)
    return norm

img = np.asarray(pil.open('cameraman.png'), dtype=np.float64)[:, :, 0]

# part b
p_values = np.linspace(1.0, 5.0, num=81)
lp_values = [lp_norm(img, x) for x in p_values]

# graphing
plt.rcParams['text.usetex'] = True
plt.plot(p_values, lp_values, linewidth=2)
plt.title(r'$L^p$ norms', fontsize=16)
plt.xlabel(r'$p$', fontsize=14)
plt.ylabel(r'norm', fontsize=14)
plt.savefig('part_b.pdf')
plt.close

# part d
print("Forward difference: ", sobolev_norm(img, 2, forward_difference))
print("Backward difference: ", sobolev_norm(img, 2, backward_difference))
print("Centered difference: ", sobolev_norm(img, 2, centered_difference))