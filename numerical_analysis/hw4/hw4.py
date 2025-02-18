import src.hw4_1
import src.hw4_2
import numpy as np
from pathlib import Path


# question 1
def question_1() -> None:

    print("\nQUESTION 1")

    # read input file
    with open(Path("inputs/input4_1.txt")) as input_file:
        lines = [x.rstrip() for x in input_file]

    # input file information
    n = int(lines[0]) # size of the matrix
    rhs = np.fromstring(lines[1], sep=",") # rhs
    x0 = np.fromstring(lines[2], sep=",") # initial guess
    tol = float(lines[3]) # tolerance level
    max_iter = int(lines[4]) # maximum iterations
    start = float(lines[5]) # starting value for part A
    end = float(lines[6]) # ending value for part A
    num_val = int(lines[7]) # number of values for part A
    eps = float(lines[8]) # epsilon value for part B

    # perform the operations in question 1
    src.hw4_1.hw4_1_main(n, rhs, x0, tol, max_iter, start, end, num_val, eps)

    return None


# question 2
def question_2() -> None:
    
    print("\nQUESTION 2\n")

    # read input file
    with open(Path("inputs/input4_2.txt")) as input_file:
        lines = [x.rstrip() for x in input_file]
    
    # input file information
    n = int(lines[0]) # size of the matrix
    rhs = np.fromstring(lines[1], sep=",") # rhs
    x0 = np.fromstring(lines[2], sep=",") # initial guess
    tol = float(lines[3]) # tolerance level
    max_iter = int(lines[4])

    # perform the operations in question 2
    src.hw4_2.hw4_2_main(n, rhs, x0, tol, max_iter)

    return None


# wrapper function
def hw4_main() -> None:
    while True:
        question_number = input("\nEnter the question number. Type an E to exit.\n")

        if question_number.lower() == "e":
            break

        elif question_number == "1":
            question_1()

        elif question_number == "2":
            question_2()

        else:
            print("Entry not valid.")

    return None


if __name__=="__main__":
    hw4_main()
