
#我觉得我应该在开一个分支提交代码 能够保证修改

import taichi as ti

from Linear_Equation_Solver import *


import sys
sys.path.append('FEM Soft Body')
from Implicit_FEM_Soft_Body import *

ti.init(arch=ti.gpu)

@ti.data_oriented
class Newton_Method:

    def __init__(self):

        self.impobj = Implicit_Object('tetrahedral-models/ellell.1', 0)
        # 把vn作为初值带入
        self.x = self.impobj.velocity
        self.F_num = self.impobj.F_num
        self.F_Jac = self.impobj.F_Jac
        self.matrixsize = self.x.shape[0]
        self.dx = ti.field(dtype=ti.f32,shape=self.matrixsize)
        self.equationsolver = Linear_Equation_Solver(self.F_Jac, self.F_num, self.dx) 
        

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

            #如果这里要写成implicit的话 大概应该是
            #compute_force()
            #compute_force_gradient()
            #assembly() 把F和F_Jac全都弄出来

            
            #这一步输入A=F_Jac dx=xn-xn+1 b=F
            #更新dx
            self.equationsolver.Jacobi(100,1e-5)
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

    @ti.kernel
    def damped_Newton(self):
        print("damped Newton")
        pass


# 好 直接带值进去现在是正常的了
newton_method = Newton_Method()
#newton_method.ordinary_Newton(100, 1e-5)





















#newton_method = Newton_Method()
#newton_method.damped_Newton()
