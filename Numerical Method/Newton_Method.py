
# Linear_Equation_Solver可以作为Newton的子类出现 直接调用就可以了
# 这个地方用damped Newton写的乱七八糟的 完全不知道在干什么
# 再检查一遍算法 看看到底是哪里不对

import taichi as ti

from Linear_Equation_Solver import Linear_Equation_Solver

ti.init(arch=ti.gpu)

@ti.data_oriented
class Newton_Method:

    def __init__(self, x: ti.template(), F:ti.template(), F_Jac:ti.template()):

        self.x = x
        self.F = F
        self.F_Jac = F_Jac
        self.matrixsize = self.x.shape[0]
        self.dx = ti.field(dtype=ti.f32,shape=self.matrixsize)
        self.x_alpha = ti.field(dtype=ti.f32,shape=self.matrixsize)
        self.equationsolver = Linear_Equation_Solver(self.F_Jac, self.F, self.dx)
        
        # stepsize alpha
        self.alpha = ti.field(dtype=ti.f32, shape=())
        self.F_temp = ti.field(dtype=ti.f32,shape=self.matrixsize)

    @ti.func
    def field_norm(self, x:ti.template()):
        result = 0.0
        for i in range(self.matrixsize):
            result += x[i]**2
        return result

    @ti.func
    def initial_stepsize(self, a:ti.f32, b:ti.f32):
        alpha = 0.0
        if b < a: alpha = b
        else: alpha = ti.min(1.0, 2.0*b)
        return alpha

    @ti.func
    def field_add(self, x_alpha:ti.template(), x:ti.template(), dx:ti.template(), alpha:ti.f32):
        for i in range(x.shape[0]):
            x_alpha[i] = x[i] + alpha * dx[i]
        return x_alpha


    @ti.kernel
    def ordinary_Newton(self, max_iter_num:ti.i32, tol:ti.f32):

        iter_i = 0
        #临时的 不需要设为全局变量
        alpha1, alpha2 = 1.0, 1.0
        mu = 0.5

        while iter_i < max_iter_num:

            TestFunction(self.F, self.x)
            TestFunctionJac(self.F_Jac, self.x)
            
            self.equationsolver.Jacobi(100, 1e-6)
            #self.equationsolver.CG(100, 1e-6)

            for i in range(self.matrixsize):
                self.dx[i] = -self.dx[i]

            norm_F = self.field_norm(self.F)
            norm_dx = self.field_norm(self.dx)
            

            self.alpha[None] = self.initial_stepsize(alpha1, alpha2)

            self.field_add(self.x_alpha, self.x, self.dx, self.alpha[None])
            TestFunction(self.F_temp, self.x_alpha)
            norm_F_temp = self.field_norm(self.F_temp)
            
            while norm_F_temp > (1.0 - mu * self.alpha[None]) * norm_F:
                self.alpha[None] *= 0.5

                self.field_add(self.x_alpha, self.x, self.dx, self.alpha[None])
                TestFunction(self.F_temp, self.x_alpha)
                norm_F_temp = self.field_norm(self.F_temp)

            alpha1 = alpha2
            alpha2 = self.alpha[None]


            for i in range(self.matrixsize):
                self.x[i] += alpha2 * self.dx[i]

            

            if norm_dx < tol or norm_F < tol:
                break

            iter_i += 1
            print(iter_i, alpha2, self.x[0], self.x[1], self.x[2])

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
        x[i] = 10.0
        F[i] = 0.1
        for j in range(3):
            F_Jac[i,j] = 0.1


newton_method = Newton_Method(x,F,F_Jac)
initialize()
newton_method.ordinary_Newton(100, 1e-8)

