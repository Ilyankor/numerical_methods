import matplotlib.pyplot as plt
import numpy as np

x, y = np.loadtxt("results.csv", delimiter=",", unpack=True)
f = y - x - y**0.85
plt.plot(f, y)
plt.show()