

import taichi as ti

from Linear_Equation_Solver import *

ti.init(arch=ti.gpu)


@ti.data_oriented
class Newton_Method:

    def __init__(self, x: ti.template(), F:ti.template(), F_Jac:ti.template()):

        self.x = x
        self.F = F
        self.F_Jac = F_Jac
        self.matrixsize = self.x.shape[0]
        self.dx = ti.field(dtype=ti.f32,shape=self.matrixsize)
        self.equationsolver = Linear_Equation_Solver(self.F_Jac, self.F, self.dx)       


    @ti.func
    def field_norm(self, x:ti.template()):
        result = 0.0
        for i in range(self.matrixsize):
            result += x[i]**2
        return result

    @ti.kernel
    def ordinary_Newton(self, max_iter_num:ti.i32, tol:ti.f32):
        iter_i = 0

        while iter_i < max_iter_num:

            #这一步是正常的没有问题
            #更新F_Jac和F
            TestFunction(self.F, self.x)
            TestFunctionJac(self.F_Jac, self.x)       
            
            #这一步输入A=F_Jac dx=xn-xn+1 b=F
            #更新dx
            self.equationsolver.Jacobi(100,1e-6)
            #self.equationsolver.CG(100,1e-5)

            #原方程是-A(dx)=b 现在解出来的是A(dx)=b 解是负数 这里dx=xn-xn+1
            for i in range(self.matrixsize):
                self.x[i] -= self.dx[i]

            norm_F = self.field_norm(self.F)
            norm_dx = self.field_norm(self.dx)

            if norm_F < tol or norm_dx < tol:
                break

            iter_i += 1
            print(iter_i)
            print(self.x[0], self.x[1], self.x[2])

    @ti.kernel
    def damped_Newton(self):
        print("damped Newton")
        pass





@ti.func
def TestFunction(F:ti.template(), x:ti.template()):
    F[0] = 3*x[0]-ti.cos(x[1]*x[2])-0.5
    F[1] = x[0]**2 - 81*(x[1]+0.1)**2+ti.sin(x[2])+1.06
    F[2] = ti.exp(-x[0]*x[1])+20*x[2]+(10*3.1415926-3)/3
    return F

@ti.func
def TestFunctionJac(F_Jac:ti.template(), x:ti.template()):
    F_Jac[0,0] = 3
    F_Jac[0,1] = x[2]*ti.sin(x[1]*x[2])
    F_Jac[0,2] = x[1]*ti.sin(x[1]*x[2])
    F_Jac[1,0] = 2*x[0]
    F_Jac[1,1] = -162*(x[1]+0.1)
    F_Jac[1,2] = ti.cos(x[2])
    F_Jac[2,0] = -x[1]*ti.exp(-x[0]*x[1])
    F_Jac[2,1] = -x[0]*ti.exp(-x[0]*x[1])
    F_Jac[2,2] = 20
    return F_Jac



x = ti.field(dtype=ti.f32, shape=3)
F = ti.field(dtype=ti.f32, shape=3)
F_Jac = ti.field(dtype=ti.f32, shape=(3,3))


@ti.kernel
def initialize():
    for i in range(3):
        x[i] = 1.1
        F[i] = 0.1
        for j in range(3):
            F_Jac[i,j] = 0.1


newton_method = Newton_Method(x,F,F_Jac)
initialize()
newton_method.ordinary_Newton(100, 1e-6)

