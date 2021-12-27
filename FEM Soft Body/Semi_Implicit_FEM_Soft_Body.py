                                        
#Problem1: rendering failure: count inside face, if open only ambient color, then everything will be fine, need to cull inside faces
#Use sparse data structure to improve
#improve semi-implicit to implicit
#add damping 


import taichi as ti #version 0.8.7
import taichi_glsl as ts

import sys
sys.path.append('Numerical Method')
from Linear_Equation_Solver import *



@ti.data_oriented
class Semi_Implicit_Object:

    def __init__(self, filename, index_start=1):

        self.v = []
        self.f = []
        self.e = []

        # read nodes from *.node file
        with open(filename+".node", "r") as file:
            vn = int(file.readline().split()[0])
            for i in range(vn):
                self.v += [ float(x) for x in file.readline().split()[1:4] ] #[x , y, z]

        # read faces from *.face file (only for rendering)
        with open(filename+".face", "r") as file:
            fn = int(file.readline().split()[0])
            for i in range(fn):
                self.f += [ int(ind)-index_start for ind in file.readline().split()[1:4] ] # triangle

        # read elements from *.ele file
        with open(filename+".ele", "r") as file:
            en = int(file.readline().split()[0])
            for i in range(en):
                self.e += [ int(ind)-index_start for ind in file.readline().split()[1:5] ] # tetrahedron

        #print(self.v)
        #print(self.f)
        #print(self.e)

        self.vn = int(len(self.v)/3)
        self.fn = int(len(self.f)/3)
        self.en = int(len(self.e)/4)
        self.dim = 3
        self.inf = 1e10

        #simulation data
        self.node = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)
        self.initial_node = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)
        self.face = ti.Vector.field(3, dtype=ti.i32, shape=self.fn)
        self.element = ti.Vector.field(4, dtype=ti.i32, shape=self.en)
        #rendering data
        self.position = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)
        self.indices = ti.field(int, len(self.f))


        # for simulation
        self.dt = 1e-4
        self.gravity = 10.0
        self.E = 1e3 # Young's modulus
        self.nu = 0.3 # Poisson's ratio: nu [0, 0.5)
        self.node_mass = 1.0
        self.mu = ti.field(dtype=ti.f32, shape=())
        self.la = ti.field(dtype=ti.f32, shape=())
        self.color = (0.99,0.75,0.89)

        self.velocity = ti.field(dtype=ti.f32, shape=self.dim*self.vn)
        self.force = ti.field(dtype=ti.f32, shape=self.dim*self.vn)
        self.element_volume = ti.field(dtype=ti.f32, shape=self.en)
        self.B = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.en) # a square matrix

        # derivatives
        self.dD = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.en, 4, 3))
        self.dF = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.en, 4, 3))
        self.dP = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.en, 4, 3))
        self.dH = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.en, 4, 3))
        self.initdD = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.en, 4, 3))
        self.initdF = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.en, 4, 3))
        self.initdP = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.en, 4, 3))
        self.initdH = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.en, 4, 3))


        # df/dx
        self.force_gradient = ti.field(dtype=ti.f32, shape=(self.dim*self.vn, self.dim*self.vn))


        # for solving system of linear equations
        self.A = ti.field(dtype=ti.f32, shape=(self.dim*self.vn, self.dim*self.vn))
        self.b = ti.field(dtype=ti.f32, shape=self.dim*self.vn)
        self.x = ti.field(dtype=ti.f32, shape=self.dim*self.vn)
        self.equationsolver = Linear_Equation_Solver(self.A, self.b, self.x)
        

        print("vertices: ", self.vn, "elements: ", self.en)

        #view node as initial position, height=y+8.0
        for i in range(self.vn):
            self.node[i] = [self.v[3*i], self.v[3*i+1]+8.0, self.v[3*i+2]] 
            self.initial_node[i] = [self.v[3*i], self.v[3*i+1]+8.0, self.v[3*i+2]]

        for i in range(self.dim*self.vn):
            self.velocity[i] = 0.0

        for i in range(self.en):
            self.element[i] = [self.e[4*i], self.e[4*i+1], self.e[4*i+2], self.e[4*i+3]]

        #To be Continued: only need boundary faces, 
        #not for calculation only for rendering
        for i in range(self.fn):
            self.face[i] = [self.f[3*i], self.f[3*i+1], self.f[3*i+2]]

        #object mesh indices
        for i in range(self.fn):
            self.indices[3*i] = self.f[3*i]
            self.indices[3*i+1] = self.f[3*i+1]
            self.indices[3*i+2] = self.f[3*i+2]


    @ti.kernel
    def initialize(self):

        for i in range(self.vn):
            self.node[i] = self.initial_node[i]
            self.velocity[3*i] = 0.0
            self.velocity[3*i+1] = 0.0
            self.velocity[3*i+2] = 0.0

        for i in range(self.en):
            D = self.compute_D(i)
            # B is const matrix
            self.B[i] = D.inverse()
            #tetrahedron volume
            self.element_volume[i] = abs(D.determinant()) / 6.0

        #initialize dD dF dP matrix
        for e in range(self.en):
            for n in range(4):
                for dim in range(self.dim):
                    for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
                        self.initdD[e, n, dim][i, j] = 0
                        self.initdF[e, n, dim][i, j] = 0
                        self.initdP[e, n, dim][i, j] = 0

            # initial dD as one 1 rest 0 and -1 for 4th point, linear FEM change for higher order
            for n in ti.static(range(3)):
                for dim in ti.static(range(self.dim)):
                    self.initdD[e, n, dim][dim, n] = 1
            for dim in ti.static(range(self.dim)):
                self.initdD[e, 3, dim] = - (self.initdD[e, 0, dim] + self.initdD[e, 1, dim] + self.initdD[e, 2, dim])

            #initialize dF as dD*B
            for n in ti.static(range(4)):
                for dim in ti.static(range(self.dim)):
                    self.initdF[e, n, dim] = self.initdD[e, n, dim] @ self.B[e]




    #FEM 1st order change higher order FEM from here
    @ti.func
    def compute_D(self, i):
        a, b, c, d = self.element[i][0], self.element[i][1], self.element[i][2], self.element[i][3]
        #ti.Matrix.cols sort vectors in columns
        return ti.Matrix.cols([self.node[a]-self.node[d], self.node[b]-self.node[d], self.node[c]-self.node[d]])

    @ti.func
    def F(self, i): # deformation gradient
        return self.compute_D(i) @ self.B[i]

    @ti.func
    def Psi(self, i): # (strain) energy density
        F = self.F(i)
        J = max(F.determinant(), 0.01)
        return self.mu[None] / 2 * ( (F @ F.transpose()).trace() - self.dim ) - self.mu[None] * ti.log(J) + self.la[None] / 2 * ti.log(J)**2

    @ti.func
    def PK1(self, i):
        F = self.F(i)
        F_inv_T = F.inverse().transpose()
        J = max(F.determinant(), 0.01)
        return self.mu[None] * (F - F_inv_T) + self.la[None] * ti.log(J) * F_inv_T


    #compute force for each vertex E_total = E_strain + E_kinetic
    @ti.kernel
    def compute_force(self, node_mass:float, gravity:float):

        #add gravity
        for i in range(self.vn):
            self.force[self.dim*i] = 0
            self.force[self.dim*i+1] = -node_mass*gravity
            self.force[self.dim*i+2] = 0

        #add elasticity
        for i in range(self.en):

            a = self.element[i][0]
            b = self.element[i][1]
            c = self.element[i][2]
            d = self.element[i][3]

            #element_force = Wi * PK1 * partial(F)/partial(x) is 3*3 matrix
            #partial(F)/partial(x) = ej * ei^T * B
            #PK1 * partial(F)/partial(x) = [Pk1@B^T]_kj 1*1 element
            #Hkj k-dim j-index 
            H = -self.element_volume[i] * (self.PK1(i) @ self.B[i].transpose())

            # 3-dim force
            for j in ti.static(range(self.dim)):
                self.force[3*a+j] += H[j, 0]
                self.force[3*b+j] += H[j, 1]
                self.force[3*c+j] += H[j, 2]
                #inner force should be 0
                self.force[3*d+j] += -(H[j, 0] + H[j, 1] + H[j, 2])

    # df/dx=dF/dx^T * dP/dF * dF/dx
    @ti.kernel
    def compute_force_gradient(self):

        #竟然是这句代码报的错误 我觉得最不可能出错的代码 真是奇怪
        for i, j in ti.ndrange(self.dim*self.vn, self.dim*self.vn):
            self.force_gradient[i, j] = 0.0

        # initialize dF/dx_ij = dD/dx*B for each 
        # notice that dF/dx_ij not equal dF/dF_ij
        for e in range(self.en):
            for n in range(4):
                for dim in range(self.dim):
                    self.dD[e, n, dim] = self.initdD[e, n, dim]
                    self.dF[e, n, dim] = self.initdF[e, n, dim]
                    self.dP[e, n, dim] = self.initdP[e, n, dim]


        #calculate element stiffness matrix of gradient force
        for e in range(self.en):

            #deformation gradient F F^-1 F^-T
            F = self.F(e)
            F_inv = F.inverse()
            F_inv_T = F_inv.transpose()
            # trick
            J = max(F.determinant(), 0.01)

            #calculate dP/dx fourth order tensor
            for n in range(4):
                for dim in range(self.dim):
                    # calculate every single number of dP/dx=dF/dx:dP/dF
                    for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
                            
                        # dF/dF_{ij} one 1.0 eight 0.0
                        dF = ti.Matrix.zero(float, 3, 3)
                        dF[i, j] = 1.0
                        # dF^T/dF_{ij}
                        dF_T = dF.transpose()
                        # Tr( F^{-1} dF/dF_{ij}) = Tr(F^-1*ei*ej) = F^-1_ij
                        dTr = F_inv[i, j]

                        #dP/dF = mu*dF+(mu-la*logJ)F^-T*dF^T+la*tr(F^-1*dF)F^-T 3*3 matrix                       
                        dP_dFij = self.mu[None] * dF + (self.mu[None] - self.la[None] * ti.log(J)) * F_inv_T @ dF_T @ F_inv_T + self.la[None] * dTr * F_inv_T

                        # inner product of dF/dx:dP/dF 
                        # need to calculate (3*3)*(3*3) to generate 3*3 dP/dx_ij
                        self.dP[e, n, dim] += dP_dFij * self.dF[e, n, dim][i, j]

                        
            # obtain derivative of Hessian matrix dH=(df1,df2,df3)^T
            for n in ti.static(range(4)):
                for dim in ti.static(range(self.dim)):
                    self.dH[e, n, dim] = -self.element_volume[e] * self.dP[e, n, dim] @ self.B[e].transpose()


            # fill the element df/dx matrix of size (dim*nodes)^2
            for n in ti.static(range(4)):
                for i in ti.static(range(self.dim)):
                    ind = self.element[e][n] * self.dim + i
                    for j in ti.static(range(self.dim)):
                        # df_{jx}/dx_{ndim} df_{jy}/dx_{ndim} df_{jz}/dx_{ndim}
                        self.force_gradient[self.element[e][j]*3+0, ind] += self.dH[e, n, i][0, j] 
                        self.force_gradient[self.element[e][j]*3+1, ind] += self.dH[e, n, i][1, j] 
                        self.force_gradient[self.element[e][j]*3+2, ind] += self.dH[e, n, i][2, j] 

                    # fourth point based on df4/dx = -df1/dx-df2/dx-df3/dx
                    self.force_gradient[self.element[e][3]*3+0, ind] += -(self.dH[e, n, i][0, 0] + self.dH[e, n, i][0, 1] + self.dH[e, n, i][0, 2])
                    self.force_gradient[self.element[e][3]*3+1, ind] += -(self.dH[e, n, i][1, 0] + self.dH[e, n, i][1, 1] + self.dH[e, n, i][1, 2])
                    self.force_gradient[self.element[e][3]*3+2, ind] += -(self.dH[e, n, i][2, 0] + self.dH[e, n, i][2, 1] + self.dH[e, n, i][2, 2])

    # (I-h2*M^-1*gradient(f(xn)) * v_n+1 = vn+h*M^-1*f(xn)
    @ti.kernel
    def assembly(self):

        # initialize A as big identity matrix A = (I - whatever)
        # size 3-dim*nodes * 3-dim*nodes
        for i, j in self.A:
            self.A[i, j] = 1.0 if i == j else 0.0

        #assemble A matrix
        for i, j in self.A:
            self.A[i,j] -= self.dt**2 / self.node_mass * self.force_gradient[i, j]


        # assemble x matrix as initial value for iteration
        # since most time v_n and v_n+1 are similar, can use v_n as initial value
        # size 3-dim*nodes
        for i in range(self.vn):
            for j in ti.static(range(self.dim)):
                self.x[i*self.dim+j] = self.dt * self.velocity[i*self.dim+j]

        # assmeble b matrix
        # size 3-dim*nodes
        for i in range(self.vn):
            for j in ti.static(range(self.dim)):
                self.b[i*self.dim+j] = self.dt * self.velocity[i*self.dim+j] + self.dt**2 / self.node_mass * self.force[i*self.dim+j]


