
import taichi as ti
import time


@ti.data_oriented
class Linear_Equation_Solver:

    def __init__(self, A:ti.template(), b:ti.template(), x:ti.template()):
        self.A = A
        self.x = x
        self.b = b
        self.matrixsize = b.shape[0]
        self.x_new = ti.field(dtype=ti.f32, shape=self.matrixsize)
        self.r = ti.field(dtype=ti.f32, shape=self.matrixsize) 
        # conjugate gradient residual
        self.d = ti.field(dtype=ti.f32, shape=self.matrixsize)
        self.q = ti.field(dtype=ti.f32, shape=self.matrixsize)

    # Jacobi iteration
    # 这个先改成pyfunc既能在python 又能在taichi里面走
    @ti.func
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

            res = 0.0 #!!!
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
    def CG(self, max_iter_num:ti.i32, tol:ti.f32): # conjugate gradient
        
        n = self.matrixsize
        
        for i in range(n): 
            r = self.b[i]*1.0
            for j in range(n):
                r -= self.A[i, j] * self.x[j]
            self.r[i] = r
            self.d[i] = r
        
        delta_new = 0.0
        for i in range(n):
            delta_new += self.r[i]*self.r[i]
        delta_0 = delta_new

        if delta_0 < tol: pass

        iter_i = 0
        while iter_i < max_iter_num:


            for i in range(n):
                r = 0.0 #!!!
                for j in range(n):
                    r += self.A[i, j] * self.d[j]
                self.q[i] = r
            
            alpha = 0.0
            for i in range(n):
                alpha += self.d[i] * self.q[i]
            alpha = delta_new / alpha

            for i in range(n):
                self.x[i] += alpha * self.d[i]


            if (iter_i+1) % 50 == 0:
                for i in range(n):
                    r = self.b[i]*1.0
                    for j in range(n):
                        r -= self.A[i, j] * self.x[j]
                    self.r[i] = r
            else:
                for i in range(n):
                    self.r[i] -= alpha * self.q[i]

            delta_old = delta_new
            delta_new = 0.0
            for i in range(n):
                delta_new += self.r[i] * self.r[i]

            if delta_new < tol: break

            beta = delta_new / delta_old

            for i in range(n):
                self.d[i] = self.r[i] + beta * self.d[i]

            iter_i += 1


