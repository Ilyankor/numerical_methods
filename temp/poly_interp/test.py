import numpy as np

# function
def func(min_in:float, max_in:float, r:np.random.Generator) -> np.ndarray:
    # degree of polynomial
    deg = r.integers(low=1, high=15)

    # coefficients
    coef = rng.random(size=(deg + 1))

    # number of inputs
    num = 0
    while num < deg:
        num = r.integers(low=3, high=1000)

    # generate input range
    int_range = r.integers(min_in, max_in, size=2, endpoint=True)
    real_min = np.min(int_range)
    real_max = np.max(int_range)

    # generate inputs
    inputs = (real_max - real_min) * r.random(size=num) + real_min

    # generate outputs with noise
    A = np.zeros((num, (deg+1)))
    for i in range(deg + 1):
        A[:, i] = inputs**i

    outputs = np.matmul(A, coef) #+ r.normal(size=num)

    return deg, num, coef, inputs, outputs

# random generator
rng = np.random.default_rng()

# create data
d, n, c, i, o = func(-30, 30, rng)

# convert to strings
i = str(i.tolist())
i = i[1:-1]

o = str(o.tolist())
o = o[1:-1]

# save file
with open("input.txt", mode="w") as file:
    file.write(f"{d}, {n}\n{i}\n{o}\n{c}")