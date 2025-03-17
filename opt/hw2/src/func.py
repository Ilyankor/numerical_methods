import numpy as np
from numpy.typing import NDArray


def prox4(
        x: NDArray[np.float64],
        rho: float,
        A: NDArray[np.float64] = None,
        L: NDArray[np.float64] = None,
        ATy: NDArray[np.float64] = None,
    ) -> NDArray[np.float64]:

    """
    Proximal operator of f for problem 4 using the Woodbury matrix identity.
    f(x) = 1/2 || Ax - y ||^2.

    Args:
        x: Argument for the proximal operator.
        rho: Proximal parameter.
        A: Matrix from f(x).
        L: Lower Cholesky factor of I + rho*A*A^T, precomputed.
        ATy: Precomputed A^T*y
    
    Returns:
        prox(rho, f)(x)
    """

    b = A @ (x + rho * ATy)
    u = np.linalg.solve(L, b)
    v = np.linalg.solve(L.T, u)
    w = x + rho * (ATy - A.T @ v)
    
    return w


def prox_fro(x: NDArray[np.float64], rho: float = 1.0, a: float = 1.0) -> NDArray[np.float64]:
    """
    Proximal operator for the scaled Frobenius norm.
    f(x) = a * || x ||_F.

    Args:
        x: Matrix input.
        rho: Proximal parameter.
        a: Scale factor.

    Returns:
        prox(rho, f)(x)
    """

    return (1.0 / (1.0 + a * rho)) * x


def prox_one(x: NDArray[np.float64], rho: float = 1.0, a: float = 1.0) -> NDArray[np.float64]:
    """
    Proximal operator for the scaled matrix 1-norm.
    Also an elementwise soft thresholding operator.
    f(x) = a * || x ||_1.

    Args:
        x: Matrix input.
        rho: Proximal parameter.
        a: Scale factor.

    Returns:
        prox(rho, f)(x)
    """

    return np.maximum(0, x - a * rho) - np.maximum(0, -x - a * rho)


def prox_nuc(x: NDArray[np.float64], rho: float = 1.0, a: float = 1.0) -> NDArray[np.float64]:
    """
    Proximal operator for the scaled nuclear norm.
    f(x) = a * || x ||_*.

    Args:
        x: Matrix input.
        rho: Proximal parameter.
        a: Scale factor.

    Returns:
        prox(rho, f)(x)
    """

    U, S, Vh = np.linalg.svd(x, full_matrices=False)
    S = prox_one(S, a, rho)

    return (U * S) @ Vh


def prox5(x: NDArray[np.float64], rho: float = 1.0, a: tuple = (1.0, 1.0, 1.0)) -> NDArray[np.float64]:
    """
    Component-wise proximal operators for problem 5.
    Inputs x = (x1, x2, x3) as a stack of matrices and applies the proper operators to each component.

    Args:
        x: Stack of 3 matrices as an (m, n, 3) NDArray.
        rho: Proximal parameter.
        a: Scale factors for each component functions.

    Returns:
        Stack of prox(rho, f)(x) as an (m, n, 3) NDArray.
    """

    prox_funcs = [prox_fro, prox_one, prox_nuc]
    y = np.zeros_like(x)
    for i, prox in enumerate(prox_funcs):
        y[..., i] = prox(x[..., i], rho, a[i])

    return y


def primalstop5(r: NDArray[np.float64], x: NDArray[np.float64], z: NDArray[np.float64]) -> bool:
    """
    Primal stopping criteria for problem 5.

    Args:
        r: Primal residuals as an (m, n, 3) NDArray, r = x - z.
        x: Primal variable for f(x).
        z: Primal variable for g(x).
    
    Returns:
        True if stopping criteria is satisfied, False if not.
    """

    atol = 1e-4     # absolute tolerance
    rtol = 1e-2     # relative tolerance
    m, n, _ = x.shape
    epspri = np.sqrt(m * n) * atol + rtol * (
        np.max(np.concatenate((np.linalg.norm(x, axis=(0, 1)), np.linalg.norm(z, axis=(0, 1)))))
    )
    rnorm = np.linalg.norm(r[..., 0])

    return rnorm < epspri


def dualstop5(s: NDArray[np.float64], u: NDArray[np.float64]) -> bool:
    """
    Dual stopping criteria for problem 5.

    Args:
        s: Dual residuals as an (m, n, 3) NDArray.
        u: Dual variable.
    
    Returns:
        True if stopping criteria is satisfied, False if not.
    """

    atol = 1e-4     # absolute tolerance
    rtol = 1e-2     # relative tolerance
    m, n, _ = u.shape
    epsdual = np.sqrt(m * n) *  atol + rtol * np.linalg.norm(1.0 * u[..., 0])
    snorm = np.max(np.linalg.norm(s, axis=(0, 1)))

    return snorm < epsdual


if __name__ == "__main__":
    exit(0)
