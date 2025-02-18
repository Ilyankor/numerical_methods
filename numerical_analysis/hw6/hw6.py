import src.hw6_1
import src.hw6_2
import src.hw6_3
import numpy as np
from pathlib import Path


# question 1
def question_1() -> None:

    print("\nQUESTION 1\n")

    # read input file
    with open(Path("inputs/input6_1.txt")) as input_file:
        lines = [x.rstrip() for x in input_file]
    
    # input file information
    n = int(lines[0]) # size of the matrix
    rhs = np.fromstring(lines[1], sep=",") # rhs
    x0 = np.fromstring(lines[2], sep=",") # initial guess
    tol = float(lines[3]) # tolerance level
    max_iter = int(lines[4])

    # perform the operations in question 1
    src.hw6_1.hw6_1_main(n, rhs, x0, tol, max_iter)

    return None


# question 2
def question_2() -> None:
    
    print("\nQUESTION 2")

    # read input files
    paths = ["input6_2_1.txt", "input6_2_2.txt", "input6_2_3.txt"]

    # store information
    info = {}
    for path in paths:
        with open(Path("inputs/" + path)) as input_file:
            lines = [x.rstrip() for x in input_file]
        
        # input file information
        n = int(lines[0]) # size of the matrix
        A_info = {
            "A": np.array([np.fromstring(x, sep=",") for x in lines[1:1+n]]),
            "q0": np.fromstring(lines[n+1], sep=","),
            "tol": float(lines[n+2]),
            "max_iter": int(lines[n+3])
        }

        # add to question 2 information dict
        info.update({"A" + path[9]: A_info})

    # perform the operations in question 1
    src.hw6_2.hw6_2_main(info)

    return None


# question 3
def question_3() -> None:

    print("\nQUESTION 3\n")

    # read input file
    with open(Path("inputs/input6_3.txt")) as input_file:
        lines = [x.rstrip() for x in input_file]
    
    # input file information
    n = int(lines[0]) # size of the matrix 2*n+1
    mu = float(lines[1]) # shift value
    q0 = np.fromstring(lines[2], sep=",") # rhs
    tol = float(lines[3]) # tolerance level
    max_iter = int(lines[4])

    # perform the operations in question 1
    src.hw6_3.hw6_3_main(n, mu, q0, tol, max_iter)

    return None


# wrapper function
def hw6_main() -> None:
    while True:
        question_number = input("\nEnter the question number. Type an E to exit.\n")

        if question_number.lower() == "e":
            break

        elif question_number == "1":
            question_1()

        elif question_number == "2":
            question_2()

        elif question_number == "3":
            question_3()

        else:
            print("Entry not valid.")

    return None


if __name__=="__main__":
    hw6_main()
