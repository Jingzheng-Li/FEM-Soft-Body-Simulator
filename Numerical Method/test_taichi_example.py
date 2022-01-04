

import numpy as np
import taichi as ti

import numpy as np
import random

ti.init(arch=ti.cpu)


# 初始化矩阵
n = 3
#A = np.array([[1,2,4],[2,13,23],[4,23,77]])


L = ti.field(dtype=ti.f32, shape=(3,3))
A = ti.field(dtype=ti.f32, shape=(3,3))

@ti.kernel
def initialize_matrix():
    A[0,0] = 1
    A[0,1] = 2
    A[0,2] = 4
    A[1,0] = 2
    A[1,1] = 13
    A[1,2] = 23
    A[2,0] = 4
    A[2,1] = 23
    A[2,2] = 77


@ti.kernel
def Cholesky_Decomposition(A:ti.template(), L:ti.template()):
    
    for i, j in ti.ndrange(A.shape[0], A.shape[0]):
        L[i,j] = A[i,j]

    for k in range(A.shape[0]):

        L[k,k] = ti.sqrt(L[k,k])

        for i in range(k+1,A.shape[0]):
            if L[i,k] != 0:
                L[i,k] /= L[k,k]

        for j in range(k+1,A.shape[0]):
            for i in range(j,A.shape[0]):
                if L[i,j] != 0:
                    L[i,j] -= L[i,k] * L[j,k]

    for i, j in ti.ndrange(A.shape[0], A.shape[0]):
        if j > i: L[i,j] = 0


@ti.kernel
def Preconditioner():
    resultTemp = np.zeros((n))
    for i in range(n):
        resultTemp[i] = r[i] / L[i,i]
        for j in range(i):
            resultTemp[i] -= L[i,j] / L[i,i] * resultTemp[j]
    # 求解 L^T * result = rhs 即 result = L^T^-1 rhs
    # backward subtitution 后向替换
    result = np.zeros((n))
    for i in range(n-1,-1,-1):
        result[i] = resultTemp[i] / L[i,i]
        for j in range(i+1,n):
            result[i] -= L[j,i] / L[i,i] * result[j]





initialize_matrix()
Cholesky_Decomposition(A, L)

for i in range(3):
    print(A[i,0],A[i,1],A[i,2])

print("============================")

for i in range(3):
    print(L[i,0],L[i,1],L[i,2])






