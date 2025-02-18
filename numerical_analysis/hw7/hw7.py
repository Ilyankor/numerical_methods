import src.hw7_1
import src.hw7_2
# import src.hw6_3
# import numpy as np
from pathlib import Path


# question 1
def question_1() -> None:

    print("\nQUESTION 1\n")

    # read input file
    with open(Path("inputs/input7_1.txt")) as input_file:
        lines = [x.rstrip() for x in input_file]
    
    # input file information
    n_val = int(lines[0]) # divisions for graphing
    a = float(lines[1]) # left endpoint of interval a
    b = float(lines[2]) # right endpoint of interval b
    tol = float(lines[3]) # tolerance level
    max_iter = int(lines[4]) # maximum iterations

    # perform the operations in question 1
    src.hw7_1.hw7_1_main(n_val, a, b, tol, max_iter)

    return None


# question 2
def question_2() -> None:
    
    print("\nQUESTION 2")

    # read input file
    with open(Path("inputs/input7_2.txt")) as input_file:
        lines = [x.rstrip() for x in input_file]
    
    # input file information
    x0 = float(lines[0]) # initial guess
    tol = float(lines[1]) # tolerance level
    max_iter = int(lines[2]) # maximum iterations
    alpha = 2**(0.25) # actual root alpha

    # perform the operations in question 2
    src.hw7_2.hw7_2_main(x0, tol, max_iter, alpha)

    return None


# wrapper function
def hw7_main() -> None:
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
    hw7_main()
