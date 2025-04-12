from src.algo import *
from src.func import *
from src.util import *
import scipy as sci

def question_2() -> None:
    print("\nQUESTION 2")

    # question number
    q = 2

    # problem parameters
    t0 = 0.0
    tn = 60.0

    p0 = np.array([20.0, 5.0])
    params = (1.0, 0.2, 0.3, 1.0)

    # algorithm time steps
    n_euler = 1000
    n_rk4 = 500
    
    # solutions
    t_euler, sol_euler = euler(predatorprey, p0, t0, tn, n_euler, param=params)
    t_rk4, sol_rk4 = rk4(predatorprey, p0, t0, tn, n_rk4, param=params)
    result_rk45 = sci.integrate.solve_ivp(predatorprey, [t0, tn], p0, method="RK45", args=(params,))

    # save solutions
    arr = {
        "t_euler": t_euler,
        "sol_euler": sol_euler,
        "t_rk4": t_rk4,
        "sol_rk4": sol_rk4,
        "t_rk45": result_rk45.t,
        "sol_rk45": result_rk45.y.T,
    }
    dirpath = Path("out/2")
    save_arrays(arr, dirpath)

    # visualization
    visualize(q, arr)

    return None


def question_7() -> None:
    print("\nQUESTION 7\n")

    # question number
    q = 7

    # problem parameters
    t0 = 0.0
    tn = 1.0

    u0 = np.array([0.98, 0.02])
    params = (26.0, 11.0)

    # algorithm time steps
    n = 1024

    # solution
    t, sol = sir(params, u0, t0, tn, n)

    # save solutions
    arr = {
        "t": t,
        "sol": sol
    }
    dirpath = Path("out/7")
    save_arrays(arr, dirpath)

    # visualization
    visualize(q, arr)

    return None


# wrapper function
def main() -> None:
    while True:
        question = input("\nEnter the question number. Type an E to exit.\n")

        match question.lower():
            case "e":
                break
            case "2":
                question_2()
            case "7":
                question_7()
            case _:
                print("Entry not valid.")

    return None


if __name__ == "__main__":
    main()
