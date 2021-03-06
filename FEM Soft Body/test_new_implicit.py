

import taichi as ti #version 0.8.7
import taichi_glsl as ts


import sys
sys.path.append('Numerical Method')
from Linear_Equation_Solver import *


@ti.data_oriented
class Implicit_Object:

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
        self.dt = 1e-3
        self.gravity = 10.0
        self.E = 5000 # Young's modulus
        self.nu = 0.3 # Poisson's ratio: nu [0, 0.5)
        self.node_mass = 1.0
        self.mu = ti.field(dtype=ti.f32, shape=())
        self.la = ti.field(dtype=ti.f32, shape=())
        self.modelscale = 0.2
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


        # for solving system of Newton iterations
        self.F_Jac = ti.field(dtype=ti.f32, shape=(self.dim*self.vn, self.dim*self.vn))
        self.F_num = ti.field(dtype=ti.f32, shape=self.dim*self.vn)
        self.dx = ti.field(dtype=ti.f32, shape=self.dim*self.vn) # dx???x-x_n
        self.equationsolver = Linear_Equation_Solver(self.F_Jac, self.F_num, self.dx)
        # record of xn Newton Method update only when calculate velocity vn+1
        self.previous_node = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)
        # record of vn Newton Method
        self.previous_velocity = ti.field(dtype=ti.f32, shape=self.dim*self.vn)


        #for damped Newton
        #self.alpha = ti.field(dtype=ti.f32, shape=())
        self.x_alpha = ti.field(dtype=ti.f32, shape=self.dim*self.vn)
        self.F_temp = ti.field(dtype=ti.f32, shape=self.dim*self.vn)
        self.current_node = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)

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

        for e in range(self.en):
            # initial dD as one 1 rest 0 and -1 for 4th point, linear FEM change for higher order
            for n in ti.static(range(3)):
                for dim in ti.static(range(self.dim)):
                    self.initdD[e, n, dim][dim, n] = 1
            for dim in ti.static(range(self.dim)):
                self.initdD[e, 3, dim] = - (self.initdD[e, 0, dim] + self.initdD[e, 1, dim] + self.initdD[e, 2, dim])

        for e in range(self.en):
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
    def compute_force(self, node_mass:ti.f32, gravity:ti.f32):

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

        for i, j in ti.ndrange(self.dim*self.vn, self.dim*self.vn):
            self.force_gradient[i, j] = 0.0

        # initialize dF/dx_ij = dD/dx*B for each
        # notice that dF/dx_ij not equal dF/dF_ij
        for e in range(self.en):
            for n, dim in ti.ndrange(4, self.dim):
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
            for n, dim in ti.ndrange(4, self.dim):
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

             
        for e in range(self.en):
            # obtain derivative of Hessian matrix dH=(df1,df2,df3)^T
            for n, dim in ti.ndrange(4, self.dim):
                self.dH[e, n, dim] = -self.element_volume[e] * self.dP[e, n, dim] @ self.B[e].transpose()

        for e in range(self.en):
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


    @ti.kernel
    def assemble_F_Jac(self):

        #assemble F_Jac matrix M - h^2*force_gradient
        for i,j in ti.ndrange(self.vn*self.dim, self.vn*self.dim):
            self.F_Jac[i,j] = 1.0 if i == j else 0.0
        
        for i,j in ti.ndrange(self.vn*self.dim, self.vn*self.dim):
            self.F_Jac[i,j] -= self.dt**2 / self.node_mass * self.force_gradient[i, j]

    @ti.kernel
    def assemble_F_num(self):

        # assmeble F_num matrix M(x-(xn+h*vn)) - h^2*force
        for i, j in ti.ndrange(self.vn, self.dim):
            self.F_num[i*self.dim+j] = (self.dt * self.previous_velocity[i*self.dim+j] + self.dt**2 / self.node_mass * self.force[i*self.dim+j])

        for i in range(self.vn):
            for j in ti.static(range(self.dim)):
                self.F_num[i*self.dim+j] += (-self.node[i][j] + self.previous_node[i][j])


    @ti.kernel
    def field_norm(self, x:ti.template()) -> ti.f32:
        result = 0.0
        for i in range(x.shape[0]):
            result += x[i]**2
        return result


    @ti.kernel
    def matrix_field_norm(self, x:ti.template()) -> ti.f32:
        result = 0.0
        for i,j in ti.ndrange(x.shape[0],x.shape[0]):
            result += x[i,j]
        return result



    @ti.func
    def fixed_Hessian(self):
        pass
   


    @ti.kernel
    def initialize_Newton(self):

        #intialize xn and vn
        for i in range(self.vn):
            for j in ti.static(range(self.dim)):
                self.previous_node[i][j] = self.node[i][j]
                self.previous_velocity[i*self.dim+j] = self.velocity[i*self.dim+j]

        # intial dx
        for i in range(self.vn):
            for j in ti.static(range(self.dim)):
                self.dx[i*self.dim+j] = self.dt * self.velocity[i*self.dim+j]

    
    @ti.kernel
    def update_ordinary_Newton_node(self):
        # update x = x + dx
        for i in range(self.vn):
            for j in ti.static(range(self.dim)):
                self.node[i][j] += self.dx[i*self.dim+j] 



    def ordinary_Newton(self, max_iter_num:ti.i32, tol:ti.f32):

        self.initialize_Newton()

        for iter_i in range(max_iter_num):

            self.compute_force(self.node_mass, self.gravity)
            self.compute_force_gradient()
            self.assemble_F_num()
            self.assemble_F_Jac()
            
            #self.equationsolver.Jacobi(100, 1e-6)
            self.equationsolver.CG(100, 1e-6)
                
            self.update_ordinary_Newton_node()

            norm_F_num = self.field_norm(self.F_num)
            norm_dx = self.field_norm(self.dx)

            if norm_F_num < tol and norm_dx < tol:
                break








    @ti.kernel
    def initialize_stepsize(self, a:ti.f32, b:ti.f32) -> ti.f32:
        alpha = 0.0
        if b < a: 
            alpha = b
        else: 
            alpha = ti.min(1.0, 2.0*b)
        return alpha

    
    @ti.kernel
    def record_current_node(self):
        for i in range(self.vn):
            for j in ti.static(range(self.dim)):
                self.current_node[i][j] = self.node[i][j]

    @ti.kernel
    def update_F_temp(self, alpha:ti.f32):

        for i in range(self.vn):
            for j in ti.static(range(self.dim)):
                self.node[i][j] += alpha * self.dx[i*self.dim+j] 

        self.compute_force(self.node_mass, self.gravity)
        self.assemble_F_num()


    @ti.kernel
    def update_damped_Newton_node(self, alpha:ti.f32):
        # update x = x + dx
        for i in range(self.vn):
            for j in ti.static(range(self.dim)):
                self.node[i][j] = self.current_node[i][j] + alpha * self.dx[i*self.dim+j]

    def damped_Newton(self, max_iter_num:ti.i32, tol:ti.f32):

        self.initialize_Newton()
        alpha, alpha1, alpha2 = 1.0, 1.0, 1.0
        mu = 0.99

        for iter_i in range(max_iter_num):

            self.compute_force(self.node_mass, self.gravity)
            self.compute_force_gradient()
            self.assemble_F_num()
            self.assemble_F_Jac()
            
            #self.equationsolver.Jacobi(100, 1e-6)
            self.equationsolver.CG(100, 1e-6)

            norm_F_num = self.field_norm(self.F_num)
            norm_dx = self.field_norm(self.dx)

            # armijo line search
            alpha = self.initialize_stepsize(alpha1, alpha2)
            self.record_current_node()
            self.update_F_temp(alpha)
            norm_F_temp = self.field_norm(self.F_num)
            # if iter_i less than 10 may cause dt too small and stuck in while for nrom_F_temp keep in unchange
            while norm_F_temp > (1.0-mu*alpha)*norm_F_num and iter_i > 10:
                alpha *= 0.5
                self.update_F_temp(alpha)
                norm_F_temp = self.field_norm(self.F_num)

            alpha1 = alpha2
            alpha2 = alpha

            self.update_damped_Newton_node(alpha2)

            if norm_F_num < tol and norm_dx < tol:
                break

            #print(iter_i, "stepsize", alpha2)