import src.hw2_1
import src.hw2_2
import src.hw2_3
import numpy as np
from pathlib import Path


# question 1
def question_1() -> None:

    print("\nQUESTION 1")

    # read input file
    with open(Path("inputs/input2_1.txt")) as input_file:
        lines = [x.rstrip() for x in input_file]
    
    # size of the matrix
    n = int(lines[0])

    # location of the epsilon (i, j)
    eps_loc = tuple(map(int, lines[1].split(",")))

    # matrix
    A = np.array([np.fromstring(x, sep=",") for x in lines[2:2+n]])

    while True:
        # prompt which x exact to use
        q1_prompt = input("\nSelect which x_ex to use.\nEnter A for [1, 1, 1], B for [log(5/2), 1, 1], or E to exit to question selection.\n")

        if q1_prompt.lower() == "e":
            break

        elif q1_prompt.lower() == "a" or q1_prompt.lower() == "b":
            src.hw2_1.hw2_1_main(A, eps_loc, q1_prompt)

        else:
            print("Entry not valid.")

    return None


# question 2
def question_2() -> None:
    
    print("\nQUESTION 2")

    # read input file
    with open(Path("inputs/input2_2.txt")) as input_file:
        lines = [x.rstrip() for x in input_file]
    
    # size of the matrix
    n = int(lines[0])

    # matrix
    A = np.array([np.fromstring(x, sep=",") for x in lines[1:1+n]])

    # rhs
    b = np.fromstring(lines[-1], sep=",")

    # perform the operations in question 2
    src.hw2_2.hw2_2_main(A, b)

    return None


# question_3
def question_3() -> None:

    print("\nQUESTION 3")

    # read input file
    with open(Path("inputs/input2_3.txt")) as input_file:
        lines = [x.rstrip() for x in input_file]

    # convert input to proper data
    data = [np.fromstring(x, sep=",") for x in lines]

    # separate the data for the thomas algorithm
    matrix = tuple(data[0:-1])
    rhs = data[-1]

    # perform the thomas algorithm for question 3
    src.hw2_3.hw2_3_main(matrix, rhs)

    return None


# wrapper function
def hw2_main() -> None:
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
    hw2_main()
