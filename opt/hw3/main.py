from src.algo import *
from src.func import *
from src.util import *
import numpy as np
import scipy as sci

import matplotlib.pyplot as plt

def question_2() -> None:
    print("\nQUESTION 2")

    # time
    t0 = 0.0
    tn = 60.0

    # initial condition
    p0 = np.array([20.0, 5.0])

    # algorithm parameters
    n_euler = 1000
    n_rk4 = 500
    
    # solutions
    t_euler, p_euler = euler(predatorprey, p0, t0, tn, n_euler)
    t_rk4, p_rk4 = rk4(predatorprey, p0, t0, tn, n_rk4)
    result = sci.integrate.solve_ivp(predatorprey, [t0, tn], p0, method="BDF")

    # visualization
    plt.plot(t_euler, p_euler)
    plt.show()
    plt.plot(t_rk4, p_rk4)
    plt.show()
    plt.plot(result.t, result.y.T)
    plt.show()

    return None


def question_7() -> None:
    print("\nQUESTION 7\n")
    
    t0 = 0.0
    tn = 1.0

    u0 = np.array([0.98, 0.02])
    n = 1024

    t, y = heun(sir, u0, t0, tn, n)
    plt.plot(t, y)
    plt.show()

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
