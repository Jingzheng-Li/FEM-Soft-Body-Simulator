
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
    def __init__(self, x: ti.template(), F:ti.template(), F_Jac:ti.template()):
        #总得知道维数是多少？这个用谁的输入？
        #这个是不是能够改成x.shape[0]
        self.x = x
        self.F = F
        self.F_Jac = F_Jac
        self.dx = ti.field(dtype=ti.f32,shape=F_Jac.shape[0])
        self.equationsolver = Linear_Equation_Solver(F_Jac, self.dx, F)


    @ti.kernel
    def ordinary_Newton(self, max_iter_num:ti.i32, tol:ti.f32):

        print("ordinary Newton")

        for i in range(max_iter_num):
            if self.F.norm() < tol:
                break
            #这一步解出了dx的值
            self.equationsolver.Jacobi(100, 1e-5)
            #这一步解出了
            self.x = self.x + self.dx
            #判断dx是不是超标
            #这个norm不用计算 只需要找到最小值就够了
            print(self.dx.norm())
            if self.dx.norm() < tol:
                break
        #这一步应该到这里就结束了 可以开始写测试函数了
        #print(self.x)

    @ti.kernel
    def damped_Newton(self):
        print("damped Newton")
        pass



x = ti.field(dtype=ti.f32, shape=3)
F = ti.field(dtype=ti.f32, shape=3)
F_Jac = ti.field(dtype=ti.f32, shape=3*3)

x[0] = 0.1
x[1] = 0.1
x[2] = -0.1


@ti.func
def TestFunction(x:ti.template(), F:ti.template()):
    F[0] = 3*x[0]-ti.cos(x[1]*x[2])-0.5
    F[1] = x[0]**2 - 81*(x[1]+0.1)**2+ti.sin(x[2])+1.06
    F[2] = ti.exp(-x[0]*x[1])+20*x[2]+(10*3.1415926-3)/3
    return F

@ti.func
def TestFunctionJac(x:ti.template(), F_Jac:ti.template()):
    F_Jac[0][0] = 3
    F_Jac[0][1] = x[2]*ti.sin(x[1]*x[2])
    F_Jac[0][2] = x[1]*ti.sin(x[1]*x[2])
    F_Jac[1][0] = 2*x[0]
    F_Jac[1][1] = -162*(x[1]+0.1)
    F_Jac[1][2] = ti.cos(x[2])
    F_Jac[2][0] = -x[1]*ti.exp(-x[0]*x[1])
    F_Jac[2][1] = -x[0]*ti.exp(-x[0]*x[1])
    F_Jac[2][2] = 20
    return F_Jac























#newton_method = Newton_Method()
#newton_method.damped_Newton()
