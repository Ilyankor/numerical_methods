import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable


def proximal_gradient(
        n: int,
        obj_f: Callable[[NDArray[np.float64]], float],
        obj_g: Callable[[NDArray[np.float64]], float],
        grad: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        prox: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
        rho: float,
        beta: float,
        x0: NDArray[np.float64],
        max_iter: int,
        abstol: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:

    """
    Implementation of the proximal gradient method with backtracking line search.
    Use to solve problems of the form: minimize f + g where f is differentiable.

    Adapted from
    OPTIK --- Optimization Toolkit
    For details see https://github.com/andreasmang/optik

    Args:
        n: Dimension of the solution.
        obj_f: Objective function f to minimize.
        obj_g: Objective function g to minimize.
        grad: Gradient of f.
        prox: Proximal operator of g with parameter rho.
        rho: Parameter for the proximal operator.
        beta: Line search parameter.
        x0: Initial guess, with n dimensions.
        max_iter: Maximum number of iterations.
        abstol: Stopping criteria for the difference between iterations.

    Returns:
        A tuple containing
            sol: Solution at each iteration.
            obj: Objective function values at each iteration.
            i: Number of iterations before stopping criteria is reached.
    """

    # initialize the variables
    sol = np.zeros((n, max_iter+1))
    sol[:, 0] = x0
    
    obj = np.zeros(max_iter+1)
    obj[0] = obj_f(x0) + obj_g(x0)

    x = x0

    # proximal gradient method
    for i in range(max_iter):

        # store the gradient of x
        grad_x = grad(x)
        
        while True:
            # proximal gradient
            z = prox(x - rho * grad_x, rho)

            # backtracking line search
            r = z - x
            bound = obj_f(x) + np.dot(grad_x, r) + (1.0 / (2.0 * rho)) * np.linalg.norm(r)**2

            if obj_f(z) <= bound:
                break

            rho *= beta
        
        # new iterate
        x = z
        sol[:, i+1] = x
        obj[i+1] = obj_f(x) + obj_g(x)

        # stopping criteria
        if abs(obj[i+1] - obj[i]) < abstol:
            break

    return sol[:i+2], obj[:i+2], i+1


def acc_proximal_gradient(
        n: int,
        obj_f: Callable[[NDArray[np.float64]], float],
        obj_g: Callable[[NDArray[np.float64]], float],
        grad: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        prox: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
        extra: Callable[[int], float],
        rho: float,
        beta: float,
        x0: NDArray[np.float64],
        max_iter: int,
        abstol: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:

    """
    Implementation of an accelerated proximal gradient method with backtracking line search.
    Use to solve problems of the form: minimize f + g where f is differentiable.

    Args:
        n: Dimension of the solution.
        obj_f: Objective function f to minimize.
        obj_g: Objective function g to minimize.
        grad: Gradient of f.
        prox: Proximal operator of g with parameter rho.
        extra: Extrapolation function for acceleration.
        rho: Parameter for the proximal operator.
        beta: Line search parameter.
        x0: Initial guess, with n dimensions.
        max_iter: Maximum number of iterations.
        abstol: Stopping criteria for the difference between iterations.

    Returns:
        A tuple containing
            sol: Solution at each iteration.
            obj: Objective function values at each iteration.
            i: Number of iterations before stopping criteria is reached.
    """

    # initialize the variables
    sol = np.zeros((n, max_iter+1))
    sol[:, 0] = x0
    
    obj = np.zeros(max_iter+1)
    obj[0] = obj_f(x0) + obj_g(x0)

    x = x0

    # proximal gradient method
    for i in range(max_iter):

        # acceleration
        y = x + extra(i) * (x - sol[:, i-1])

        # store the gradient of y
        grad_y = grad(y)

        while True:
            # proximal gradient
            z = prox(y - rho * grad_y, rho)

            # backtracking line search
            r = z - y
            bound = obj_f(y) + np.dot(grad_y, r) + (1.0 / (2.0 * rho)) * np.linalg.norm(r)**2

            if obj_f(z) <= bound:
                break

            rho *= beta
        
        # new iterate
        x = z
        sol[:, i+1] = x
        obj[i+1] = obj_f(x) + obj_g(x)

        # stopping criteria
        if abs(obj[i+1] - obj[i]) < abstol:
            break

    return sol[:i+2], obj[:i+2], i+1


def admm(
        shape: tuple[int, ...],
        prox_f: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
        prox_g: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
        rho: float,
        x0: NDArray[np.float64],
        z0: NDArray[np.float64],
        u0: NDArray[np.float64],
        max_iter: int,
        primal_stop: Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], bool],
        dual_stop: Callable[[NDArray[np.float64], NDArray[np.float64]], bool],
        var: bool = False,
        **kwargs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:

    """
    Implementation of an alternating direction method of multipliers.
    Use to solve problems of the form: minimize f(x) + g(z) subject to x - z = 0.

    Args:
        shape: Shape of the solution.
        prox_f: Proximal operator of f with parameter rho.
        prox_g: Proximal operator of g with parameter rho.
        rho: Parameter for the proximal operators.
        x0: Initial guess for x, dimension n.
        z0: Initial guess for z, dimension n.
        u0: Initial guesses for u, dimension n.
        max_iter: Maximum number of iterations.
        primal_stop: Stopping criteria for the primal variables, a function that returns True if criteria is met.
        dual_stop: Stopping criteria for the dual variables, a function that returns True if criteria is met.
        var: Boolean indicating whether it is ADMM or the variant.
        **kwargs: Various parameters required by prox_f.

    Returns:
        A tuple containing
            solx, solz, solu: Solution at each iteration for the primal and dual variables.
            i: Number of iterations before stopping criteria is reached.
    """

    # initialize the variables
    solx = np.zeros((shape + (max_iter+1,)))
    solz = np.zeros((shape + (max_iter+1,)))
    solu = np.zeros((shape + (max_iter+1,)))
    solx[..., 0], solz[..., 0], solu[..., 0] = x0, z0, u0
    x, z, u = np.copy(x0), np.copy(z0), np.copy(u0)

    # alternating direction method of multipliers
    for i in range(max_iter):

        # proximal update of primal variables
        x = prox_f(z - u, rho, **kwargs)

        # variant of ADMM
        if var:
            z = prox_g(x, rho)
        else:
            z = prox_g(x + u, rho)

        r = x - z                       # primal residual
        s = -rho * (z - solz[..., i])   # dual residual
        u += r                          # update dual variable
        
        # save the updates
        solx[..., i+1], solz[..., i+1], solu[..., i+1] = x, z, u

        if primal_stop(r, x, z) and dual_stop(s, u):
            break
    
    return solx[..., :i+2], solz[..., :i+2], solu[..., :i+2], i+1


if __name__ == "__main__":
    exit(0)
