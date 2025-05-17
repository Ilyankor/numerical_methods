import numpy as np
from misc.zeros import *


g = lambda x: 0.25 * x**3 - 3.0 * x**2 - x + 8.0
dg = lambda x: 0.75 * x**2 - 6.0 * x - 1.0

print( f"Bisection: {bisection(g, 0.0, 2.0)}" )
print( f"Newton: {newton(g, dg, 1.0)}" )
print( f"Secant: {secant(g, 0.0, 2.0)}" )
print( f"Regula falsi: {regula_falsi(g, 0.0, 2.0)}" )

g = lambda x: np.exp(x) - x - 1.0
dg = lambda x: np.exp(x) - 1.0
d2g = lambda x: np.exp(x)

# g = lambda x: (x - 1.0) ** 2
# dg = lambda x: 2.0 * x - 2.0
# d2g = lambda x: 2.0
print( f"Modified Newton: {modified_newton(g, dg, d2g, 1.0)}" )
