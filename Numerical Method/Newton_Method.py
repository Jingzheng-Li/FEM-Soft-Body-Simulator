
#这个函数从设计之初就先想好变量是什么 什么是要输入的 什么是要输出的
#这是数据处理的失败 F是一个tensor 而F_Jac是一个matrix 这个设计之初就有问题
#看看能不能把force从tensor设计回vector
#但是重新设计有点麻烦了 我可以单独设计一个vector做计算 不知道速度如何
#这些都是后话 首先先把newton method做好 就是用矩阵和vector来做 想好了
#或者说我可以在NewtonMethod里面做一个维数替换的处理

#我明白了 把所有f和v都设计成一列的field 然后算就好了

#这里的F应该是个函数变量吗？
#还是应该只要一个常数就够了 我懂了 这里要的是个变量？
#确实没错 这里的F和F_Jac是个变量
#等下在考虑这个事情
#先把newton method写好再说
#先把Jacobian去掉 一点一点的弄 看看到底是怎么回事

#Newton Method 调试完成了 现在是正确的Newton Method 
# 可以直接调用了



import taichi as ti

from Linear_Equation_Solver import *

ti.init(arch=ti.gpu)


@ti.data_oriented
class Newton_Method:
    
    # F(x)和F'(x)是需要输入的 然后定义一个x用来输出
    # 这里F是一个向量 F_Jac是一个矩阵
    # 不对 我的x需要一个初值v 这个v怎么设计？
    # 这里的x是vn作为初值来用
    # 不对 我怀疑这里就不能这么做了 因为F和F_Jac都不再是一个常数了
    # 我怀疑整个设计思路都需要修改
    # 连带着整个设计都有点问题
    # 我想要把newton封装出来 这个是不是可行的？
    # 是不是其实根本就是不行的？必须要设计在里面？
    # 怎么没有testfunction的地方
    # 下面就是速度的问题 Newton method能不能更快一点？ 再快一点
    # 总共有三次循环 哪些是可以并行化的

    def __init__(self, x: ti.template(), F:ti.template(), F_Jac:ti.template()):
        #总得知道维数是多少？这个用谁的输入？
        #这个是不是能够改成x.shape[0]
        self.x = x
        self.F = F
        self.F_Jac = F_Jac
        self.matrixsize = self.x.shape[0]
        self.dx = ti.field(dtype=ti.f32,shape=self.matrixsize)
        self.equationsolver = Linear_Equation_Solver(self.F_Jac, self.dx, self.F)       


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
        x[i] = 0.1
        F[i] = 0.1
        for j in range(3):
            F_Jac[i,j] = 0.1


newton_method = Newton_Method(x,F,F_Jac)
initialize()
newton_method.ordinary_Newton(100, 1e-5)





















#newton_method = Newton_Method()
#newton_method.damped_Newton()
