import numpy as np
from numpy.typing import NDArray


def predatorprey(_, u: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Predator-prey model with parameters a, b, g, d:

    p1' = a*p1 - b*p1*p2
    p2' = d*p1*p2 - g*p2

    Args:
        u: The current populations.

    Returns:
        The right hand side of the model ODE.
    """

    # parameters
    a = 1.0
    b = 0.2
    g = 0.3
    d = 1.0

    # the current populations
    u0 = u[0]
    u1 = u[1]

    return np.array([
        a * u0 - b * u0 * u1, 
        d * u0 * u1 - g * u1
    ])


def sir(_, u: NDArray[np.float64]) -> NDArray[np.float64]:
    w0 = 26.0
    w1 = 11.0

    u0 = u[0]
    u1 = u[1]

    return np.array([
        -w0 * u0 * u1,
        w0 * u0 * u1 - w1 * u1
    ])


if __name__ == "__main__":
    exit(0)
