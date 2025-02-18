# Homework 6

## Homework usage

Ensure that the files are in the following file tree structure:

```text
.
├── inputs/
│   ├── input6_1.txt
│   ├── input6_2_1.txt
│   ├── input6_2_2.txt
│   ├── input6_2_3.txt
│   └── input6_3.txt
├── src/
│   ├── hw6_1.py
│   ├── hw6_2.py
│   └── hw6_3.py
├── hw6.py
└── README.MD
```

The only dependency is NumPy.
The specific code for each question is located inside `src`.

Steps:

1. To start, run `hw6.py`.
2. To select which question should be executed, type in `1`, `2`, or `3`.
3. To exit the script, type `e` or `E`.

### Question 1

The main code for question 1 is located at `src/hw6_1.py`.
It contains three functions:

- `hilbert`: constructs an $n \times n$ Hilbert matrix
- `conjugate_gradient_method`: performs the conjugate gradient method with stopping criterion
$$\frac{\lVert \mathbf{r}^{(k + 1)} \rVert_2}{\lVert \mathbf{r}^{(0)} \rVert_2} < \texttt{tol}$$
- `hw6_1_main`: controls the operations for question 1

The input file `inputs/input6_1.txt` should not be altered.

---

The conjugate gradient method took $7$ iterations to converge.

As seen in the previous homework, the gradient method with no preconditioner did not converge in less than $500000$ iterations.
This is due to the large condition number of the matrix, causing slow convergence.

By contrast, the conjugate gradient method converged a lot faster; by Theorem 4.11 in the book, since this method uses conjugate directions, in exact arithmetic the solution would have been obtained within $5$ iterations.
Due to rounding errors, it took slightly more to reach the required tolerance level, but still significantly less than the gradient method.

### Question 2

Within the prompt for question 2, type in `A1`, `A2`, or `A3` to select the matrix (respectively $A_1$, $A_2$, and $A_3$).

The main code for question 2 is located at `src/hw6_2.py`.
It contains two functions:

- `power`: performs the power method with stopping criterion
$$\left\lVert A\mathbf{q}^{(k)} - \nu^{(k)}\mathbf{q}^{(k)} \right\rVert < \texttt{tol}$$
- `hw6_2_main`: controls the operations for executing question 2

The input files `inputs/input6_2_1.txt`, `inputs/input6_2_2.txt`, and `inputs/input6_2_3.txt` should not be altered.

---

#### $A_1$

The eigenvalue with the largest modulus is $\lambda \approx 2$.
The power method took $35$ iterations to converge.

As a reference, the eigenvalues computed by NumPy are

```text
[ 0. -1.  2.]
```

#### $A_2$

The eigenvalue with the largest modulus is $\lambda \approx 2$.
The power method took $462$ iterations to converge.

As a reference, the eigenvalues computed by NumPy are

```text
[ 0.  -1.9  2. ]
```

Notice that since the eigenvalus with the second largest in modulus is $-1.9$, the ratio $\left\lvert\frac{-1.9}{2}\right\rvert$ is larger than $\left\lvert\frac{-1}{2}\right\rvert$ and is closer to $1$, the method converges slower for $A_2$ than for $A_1$, as indicated by Theorem 5.6.

#### $A_3$

For $A_3$, the power method did not converge within $10,000$ iterations.

As a reference, the eigenvaluers computed by NumPy are

```text
[0.+0.j 0.+1.j 0.-1.j]
```

Observe that there are two eigenvalues with the same modulus that are the largest, $\mathrm{i}$ and $-\mathrm{i}$.
Thus, the assumptions for the power method are not satisfied, so it is expected to not work, as demonstrated.

### Question 3

The main code for question 3 is located at `src/hw6_3.py`.
It contains five functions:

- `lu_pivot_factor`: LU factorizes a matrix by Gaussian elimination with partial pivoting
- `lu_solve`: solves $(LU) \mathbf{x} = \mathbf{b}$ given an appropriate LU factorization by forward and backward substitution
- `power_shift`: performs the power method with shift and stopping criterion
$$\frac{\lVert \mathbf{r}^{(k + 1)} \rVert_2}{\lVert \mathbf{r}^{(0)} \rVert_2} < \texttt{tol}$$
It uses an LU factorization to reduce the cost of solving the system $\left(A - \mu I \right)\mathbf{z}^{(k)} = \mathbf{q}^{(k-1)}$.
- `wilkinson`: constructs a $(2n+1) \times (2n+1)$ Wilkinson matrix
- `hw6_3_main`: controls the operations for question 3

The input file `inputs/input6_3.txt` should not be altered.

---

To use this method, $\mu$ was chosen to be $-100$ and $\mathbf{q}^{(0)} = \mathbf{e}_1$.

The negative eigenvalue with the largest modulus is $\lambda \approx -1.12488542$.
It took $1772$ iterations to converge.
