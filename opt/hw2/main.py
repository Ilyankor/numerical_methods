from src.algo import *
from src.func import *
from src.util import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def question_4() -> None:
    print("\nQUESTION 4")

    # question number
    q = 4

    # random generator
    rng = np.random.default_rng()

    # problem parameters
    m = 500         # rows
    n = 2500        # columns
    nnz = 100       # number of nonzero entries of x_true
    noise = 1e-3    # variance of noise distribution

    # create the arrays
    arrays = create_arrays(q, m, n, nnz, noise, rng=rng)
    A, _, y = arrays.values()

    # save the arrays
    dirpath = Path("in/4")
    save_arrays(arrays, dirpath)

    # compute AT, AT*y, AT*A
    AT = A.T
    ATy = AT @ y
    ATA = AT @ A

    # compute alpha
    alpha_max = np.linalg.norm(ATy, ord=np.inf)
    alpha = 0.1 * alpha_max

    # algorithm parameters
    rho = 1.0           # proximal parameter
    beta = 0.5          # line search parameter
    x0 = np.zeros(n)    # initial guess for x
    max_iter = 100      # maximum number of iterations
    atol = 1e-4         # absolute stopping criteria
    rtol = 1e-2         # relative stoping criteria
    natol = np.sqrt(n) * atol   # sqrt(n)*atol

    # compute Cholesky factorization of I + rho*A*AT
    L = np.linalg.cholesky(np.eye(m) + rho * A @ AT)

    # functions
    f = lambda u: 0.5 * np.linalg.norm(A @ u - y)**2                    # f(x)
    g = lambda u: alpha * np.linalg.norm(u, 1)                          # g(x)
    gradf = lambda u: ATA @ u - ATy                                     # grad(f)
    proxg = lambda u, kappa: (                                          # prox(g)
        np.maximum(0, u - kappa * alpha) 
        - np.maximum(0, -u - kappa * alpha)
    )
    w = lambda u: u / (u + 3)                                           # extrapolation function
    proxf1 = lambda u, kappa: (                                         # prox(f) with dgesv
        np.linalg.solve(np.eye(n) + kappa * ATA, u + kappa * ATy)
    )
    proxf2 = prox4                                                      # prox_f with Cholesky
    primalstop = lambda r, x, z: (                                      # primal stopping criteria
        np.linalg.norm(r) <= 
            (natol + rtol* np.maximum(np.linalg.norm(x), np.linalg.norm(z))))
    dualstop = lambda s, u: (                                           # dual stopping criteria
        np.linalg.norm(s) <= natol + rtol * np.linalg.norm(rho * u)
    )

    # proximal gradient
    _, result_pg, _ = proximal_gradient(n, f, g, gradf, proxg, rho, beta, x0, max_iter, atol)

    # accelerated proximal gradient
    _, result_apg, _ = acc_proximal_gradient(n, f, g, gradf, proxg, w, rho, beta, x0, max_iter, atol)

    # alternating direction method of multipliers
    solx, solz, _, iter_admm = admm((n,), proxf1, proxg, rho, x0, x0, x0, max_iter, primalstop, dualstop)
    result_admm1 = [f(solx[:, i]) + g(solz[:, i]) for i in range(iter_admm+1)]

    # alternating direction method of multipliers with Cholesky factorization
    solx, solz, _, iter_admm = admm((n,), proxf2, proxg, rho, x0, x0, x0, max_iter, primalstop, dualstop, A=A, L=L, ATy=ATy)
    result_admm2 = [f(solx[:, i]) + g(solz[:, i]) for i in range(iter_admm+1)]         

    # save solutions
    arr = {
        "sol_pg": result_pg,
        "sol_apg": result_apg,
        "sol_admm_dgesv": result_admm1,
        "sol_admm_chol": result_admm2
    }
    dirpath = Path("out/4")
    save_arrays(arr, dirpath)

    # visualize the objective functions
    arr = {
        "proximal gradient": result_pg,
        "accel. proximal gradient": result_apg,
        "ADMM": result_admm2
    }
    visualize(q, arr)

    return None


def question_5() -> None:
    print("\nQUESTION 5\n")

    # question number
    q = 5

    # random generator
    rng = np.random.default_rng()

    # problem parameters
    m = 20              # rows
    n = 50              # columns
    k = 3               # decompose into 3 matrices
    r = 4               # rank of L
    nnz = 50            # number of nonzero entries in S
    values = [-10, 10]  # values of S
    noise = 1e-3        # variance of distribution of entries of V

    # create the arrays
    arrays = create_arrays(q, m, n, nnz, noise, r, values, rng)
    L, S, V = arrays.values()
    A = L + S + V

    # save the arrays
    dirpath = Path("in/5")
    save_arrays(arrays, dirpath)

    # store 1/k * A
    kA = (1.0 / k) * A

    # algorithm parameters
    shape = (m, n, k)       # solution shape (3 mxn matrices)
    rho = 1.0               # proximal parameter
    x0 = np.zeros(shape)    # initial guess for primal x
    z0 = np.stack([kA, kA, kA], axis=2)     # initial guess for primal z
    u0 = np.zeros(shape)    # initial guess for dual u
    max_iter = 100          # maximum number of iterations

    # regularization weights
    a0 = 1.0
    a1 = 0.15 * np.max(np.abs(A))
    a2 = 0.15 * np.linalg.norm(A, ord=2)
    a = (a0, a1, a2)

    # functions (defined in src/func)
    proxf = prox5
    proxg = lambda y, _: y + np.expand_dims(kA - np.average(y, axis=2), 2)
    primalstop = primalstop5
    dualstop = dualstop5

    # alternating directions method of multipliers
    result, _, _, _ = admm(shape, proxf, proxg, rho, x0, z0, u0, max_iter, primalstop, dualstop, var=True, a=a)

    # save solution
    arr = {"sol_L": result[:, :, 2, :], "sol_S": result[:, :, 1, :], "sol_V": result[:, :, 0, :]}
    dirpath = Path("out/5")
    save_arrays(arr, dirpath)

    # evaluate the objective function
    # obj = np.zeros(it+1)
    # for i in range(it+1):
    #     obj[i] = (a0 * np.linalg.norm(x[..., 0, i], ord="fro")) + (
    #         a1 * np.linalg.norm(x[..., 1, i], ord=1)) + (
    #         a2 * np.linalg.norm(x[..., 2, i], ord="nuc"))

    # visualize the matrices
    arr = {
        r"$L$": L, 
        r"$S$": S, 
        r"$V$": V, 
        r"$\tilde{L}$": result[:, :, 2, -1], 
        r"$\tilde{S}$": result[:, :, 1, -1], 
        r"$\tilde{V}$": result[:, :, 0, -1]
    }
    visualize(q, arr)

    return None


# wrapper function
def main() -> None:
    while True:
        question = input("\nEnter the question number. Type an E to exit.\n")

        match question.lower():
            case "e":
                break
            case "4":
                question_4()
            case "5":
                question_5()
            case _:
                print("Entry not valid.")

    return None


if __name__ == "__main__":
    main()
