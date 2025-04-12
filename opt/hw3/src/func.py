import numpy as np
from .algo import heun
from numpy.typing import NDArray


def predatorprey(
        _,
        u: NDArray[np.float64],
        param: tuple[float, float, float, float]
    ) -> NDArray[np.float64]:
    
    """
    Predator-prey model with parameters a, b, g, d:

    p1' = p1*(a - b*p2)
    p2' = p2*(d*p1 - g)

    Args:
        u: The current populations.
        param: a, b, g, d.

    Returns:
        The right hand side of the model ODE.
    """

    # parameters
    a, b, g, d = param

    # the current populations
    u0 = u[0]
    u1 = u[1]

    return np.array([
        u0 * (a - b * u1), 
        u1 * (d * u0 - g)
    ])


def sirode(
        _,
        u: NDArray[np.float64],
        param: tuple[float, float]
    ) -> NDArray[np.float64]:

    """
    SIR ODE with parameters b, g:

    s' = -b*s*i
    i' = b*s*i - g*i

    Args:
        u: The current population of susceptible and infected individuals.
        param: (b, g), where b is transmission rate and g is recovery rate.

    Returns:
        The right hand side of the model ODE.
    """

    # parameters
    b, g = param

    # the current populations
    u0 = u[0]
    u1 = u[1]

    return np.array([
        -b * u0 * u1,
        b * u0 * u1 - g * u1
    ])


def sir(
        bg: tuple[float, float],
        u0: NDArray[np.float64],
        t0: float,
        tn: float,
        n: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64], NDArray[np.float64]]:

    """
    SIR model with parameters b, g:

    Solves the below ODE with Heun's method.
    s' = -b*s*i
    i' = b*s*i - g*i

    Additionally, r = 1 - s - i

    Args:
        bg: (b, g), where b is transmission rate and g is recovery rate.
        u0: The initial susceptible and infected populations.
        t0: Initial time.
        tn: Final time.
        n: Number of time steps, for Heun's method.

    Returns:
        Arrays for time, susceptible, infected, and recovered populations.
    """

    t, y = heun(sirode, u0, t0, tn, n, param=bg)
    r = 1.0 - y[:, 0] - y[:, 1]
    sol = np.column_stack([y, r])

    return t, sol


if __name__ == "__main__":
    exit(0)
