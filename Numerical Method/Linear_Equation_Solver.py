
# 今天任务 把equation solver解决好 该单独拿出来写的都拿出来
# 今天任务 把cholesky分解与条件做好


import taichi as ti
import time


@ti.data_oriented
class Linear_Equation_Solver:

    def __init__(self, A:ti.template(), b:ti.template(), x:ti.template()):
        self.A = A
        self.x = x
        self.b = b
        
        self.matrixsize = x.shape[0]
        self.x_new = ti.field(dtype=ti.f32, shape=self.matrixsize)
        self.r = ti.field(dtype=ti.f32, shape=self.matrixsize) 
        # conjugate gradient residual
        self.d = ti.field(dtype=ti.f32, shape=self.matrixsize)
        self.Ax = ti.field(dtype=ti.f32, shape=self.matrixsize) 
        self.Ad = ti.field(dtype=ti.f32, shape=self.matrixsize)
        # preconditioner conjugate gradient
        self.L = ti.field(dtype=ti.f32, shape=(self.matrixsize, self.matrixsize))
        self.z = ti.field(dtype=ti.f32, shape=self.matrixsize)
        self.z_temp = ti.field(dtype=ti.f32, shape=self.matrixsize)


    # x dot y
    @ti.func
    def dot(self, x1, x2):
        result = 0.0
        for i in range(x1.shape[0]):
            result += x1[i] * x2[i]
        return result

    # A @ x
    @ti.func
    def A_mult_x(self, Ax ,A, x):
        for i in range(x.shape[0]):
            Ax[i] = 0.0
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                Ax[i] += A[i,j] * x[j]
        return Ax

    # Jacobi stop updating
    @ti.kernel
    def Jacobi(self, max_iter_num:ti.i32, tol:ti.f32):
        n = self.matrixsize
        iter_i = 0
        res = 0.0
        while iter_i < max_iter_num:

            for i in range(n): # every row
                r = self.b[i]*1.0
                for j in range(n): # every column
                    if i != j:
                        r -= self.A[i, j] * self.x[j]
                self.x_new[i] = r / self.A[i, i]

            for i in range(n):
                self.x[i] = self.x_new[i]

            res = 0.0
            for i in range(n):
                r = self.b[i]*1.0
                for j in range(n):
                    r -= self.A[i, j] * self.x[j]
                res += r*r

            if res < tol:
                break

            iter_i += 1
        #print("Jacobi iteration:", iter_i, res)




    @ti.func
    def initialize_CG(self) -> ti.f32:
        for i in range(self.matrixsize):
            result = 0.0 
            for j in range(self.matrixsize):
                result += self.A[i, j] * self.x[j]
            self.Ax[i] = result

        for i in range(self.matrixsize):
            self.r[i] = self.b[i] - self.Ax[i]
        for i in range(self.matrixsize):
            self.d[i] = self.r[i]       
        delta_new = self.dot(self.r, self.r)
        return delta_new
    


    @ti.func
    def iteration_CG(self, delta_new:ti.f32) -> ti.f32:
        for i in range(self.matrixsize):
            result = 0.0 
            for j in range(self.matrixsize):
                result += self.A[i, j] * self.d[j]
            self.Ad[i] = result
            
        alpha = delta_new / self.dot(self.d, self.Ad)

        for i in range(self.matrixsize):
            self.x[i] += alpha * self.d[i]
        for i in range(self.matrixsize):
            self.r[i] -= alpha * self.Ad[i]

        delta_old = delta_new
        delta_new = self.dot(self.r, self.r)
        beta = delta_new / delta_old
        for i in range(self.matrixsize):
            self.d[i] = self.r[i] + beta * self.d[i]
        return delta_new


    @ti.kernel
    def CG(self, max_iter_num:ti.i32, tol:ti.f32): 
        delta_new = self.initialize_CG()
        iter_i = 0
        while iter_i < max_iter_num:
            delta_new = self.iteration_CG(delta_new)
            if delta_new < tol: break
            iter_i += 1



    # can solve the equation but severely slow down the effciency
    @ti.func
    def Incomplete_Cholesky_Factorization(self, A:ti.template(), L:ti.template()):

        for i, j in ti.ndrange(A.shape[0], A.shape[0]):
            L[i,j] = A[i,j]

        for k in range(A.shape[0]):
            L[k,k] = ti.sqrt(L[k,k])
            for i in range(k+1, A.shape[0]):
                if L[i,k] != 0:
                    L[i,k] /= L[k,k]
            for j in range(k+1, A.shape[0]):
                for i in range(j, A.shape[0]):
                    if L[i,j] != 0:
                        L[i,j] -= L[i,k] * L[j,k]

        for i, j in ti.ndrange(A.shape[0], A.shape[0]):
            if j > i: L[i,j] = 0


    @ti.func
    def ICC_Preconditioner(self):

        self.Incomplete_Cholesky_Factorization(self.A, self.L)

        for i in range(self.matrixsize):
            self.z[i] = 0.0
            self.z_temp[i] = 0.0

        for i in range(self.matrixsize):
            self.z_temp[i] = self.r[i] / self.L[i,i]
            for j in range(i):
                self.z_temp[i] -= self.L[i,j] / self.L[i,i] * self.z_temp[j]

        for i in range(self.matrixsize):
            k = (self.matrixsize-1)-i
            self.z[k] = self.z_temp[k] / self.L[k,k]
            for j in range(k+1, self.matrixsize):
                self.z[k] -= self.L[j,k] / self.L[k,k] * self.z[j]


    @ti.func
    def initialize_PCG(self) -> ti.f32:

        for i in range(self.matrixsize):
            result = 0.0 
            for j in range(self.matrixsize):
                result += self.A[i, j] * self.x[j]
            self.Ax[i] = result
        for i in range(self.matrixsize):
            self.r[i] = self.b[i] - self.Ax[i]
        
        #update self.z here
        self.ICC_Preconditioner()

        for i in range(self.matrixsize):
            self.d[i] = self.z[i]       
        delta_new = self.dot(self.r, self.z)
        return delta_new
    


    @ti.func
    def iteration_PCG(self, delta_new:ti.f32) -> ti.f32:
        for i in range(self.matrixsize):
            result = 0.0 
            for j in range(self.matrixsize):
                result += self.A[i, j] * self.d[j]
            self.Ad[i] = result
            
        alpha = delta_new / self.dot(self.d, self.Ad)

        for i in range(self.matrixsize):
            self.x[i] += alpha * self.d[i]
        for i in range(self.matrixsize):
            self.r[i] -= alpha * self.Ad[i]

        #update self.z here
        self.ICC_Preconditioner()


        delta_old = delta_new
        delta_new = self.dot(self.r, self.z)
        beta = delta_new / delta_old

        for i in range(self.matrixsize):
            self.d[i] = self.z[i] + beta * self.d[i]

        return delta_new



    @ti.kernel
    def ICC_PCG(self, max_iter_num:ti.i32, tol:ti.f32):
        #incomplete Cholesky factoriztion preconditioner
        delta_new = self.initialize_PCG()
        iter_i = 0
        while iter_i < max_iter_num:
            delta_new = self.iteration_PCG(delta_new)
            if delta_new < tol: break
            iter_i += 1


