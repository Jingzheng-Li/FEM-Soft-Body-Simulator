import taichi as ti

ti.init(arch=ti.gpu)

@ti.func
def fun1():
    print("test1")

@ti.kernel
def fun2(fun):
    fun

fun2(fun1)



