import numpy as np

def qr_iteration(A:np.ndarray, tol:float, max_iter:int) -> tuple:
    T_old = A
    for i in range(1, max_iter+1):
        Q, R = np.linalg.qr(T_old)
        T_new = np.linalg.matmul(R, Q)
        if np.linalg.norm(T_new - T_old) < tol:
            break
        T_old = T_new
    
    return np.diag(T_new), i


A = np.array([
    [1, 2, 3],
    [3, 2, 1],
    [0, 0, 1]
])

result = qr_iteration(A, 1e-10, 1000)
print(result)

res_eig = np.linalg.eigvals(A)
print(res_eig)