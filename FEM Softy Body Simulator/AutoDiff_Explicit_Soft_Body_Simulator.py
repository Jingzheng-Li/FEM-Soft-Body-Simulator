                                                                                   

import taichi as ti #version 0.8.7
import taichi_glsl as ts
import math

ti.init(arch=ti.gpu)

@ti.data_oriented
class Floor:

    #To draw mesh, need two main parameters position  & indices
    #face(0)->position 0,1,2->indices 0,1,2
    #face(1)->position 3,4,5->indices 3,4,5
    #......
    #face(N)->3N,3N+1,3N+2->indices 3N,3N+1,3N+2

    def __init__(self, height, scale):
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.indices = ti.field(int, shape=2*3)
        self.height = height
        self.scale = scale


    @ti.kernel
    def initialize(self):
        h = self.height
        k = self.scale
        self.position[0] = [-k, h, -k]
        self.position[1] = [k, h, -k]
        self.position[2] = [k, h, k]
        self.position[3] = [-k, h, k]

        N = 2
        for i, j in ti.ndrange(N, N):
            if i < N - 1 and j < N - 1:
                square_id = (i * (N - 1)) + j
                # 1st triangle of the square
                self.indices[square_id * 6 + 0] = i * N + j
                self.indices[square_id * 6 + 1] = i * N + (j + 1)
                self.indices[square_id * 6 + 2] = (i + 1) * N + j
                # 2nd triangle of the square
                self.indices[square_id * 6 + 3] = (i + 1) * N + j
                self.indices[square_id * 6 + 4] = (i + 1) * N + j + 1
                self.indices[square_id * 6 + 5] = i * N + j


@ti.data_oriented
class Object:
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
        self.indices = ti.field(int, len(self.f))
        self.dim = 3
        self.inf = 1e10


        self.node = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn, needs_grad=True)
        self.position = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)
        self.face = ti.Vector.field(3, dtype=ti.i32, shape=self.fn)
        self.element = ti.Vector.field(4, dtype=ti.i32, shape=self.en)


        ## for simulation
        self.dt = 1e-4

        self.E = 1000 # Young modulus
        self.nu = 0.3 # Poisson's ratio: nu [0, 0.5)  
        self.gravity = 10.0
        self.mu = self.E / (2 * (1 + self.nu))
        self.la = self.E * self.nu / ((1 + self.nu) * (1 -2 * self.nu))
        
        self.velocity = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)
        self.force = ti.Vector.field(self.dim,dtype=ti.f32,shape=self.vn)
        self.neighbor_element_count = ti.field(dtype=ti.i32, shape=self.vn)
        self.node_mass = ti.field(dtype=ti.f32, shape=self.vn)
        self.element_mass = ti.field(dtype=ti.f32, shape=self.en)
        self.element_volume = ti.field(dtype=ti.f32, shape=self.en)
        self.B = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.en) # a square matrix
        self.energy = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


        print("vertices: ", self.vn, "elements: ", self.en)

        for i in range(self.vn):
            self.node[i] = [self.v[3*i], self.v[3*i+1]+8.0, self.v[3*i+2]]
            self.velocity[i] = [0.0, -10.0, 0.0]

        for i in range(self.fn):
            self.face[i] = [self.f[3*i], self.f[3*i+1], self.f[3*i+2]]
            self.indices[3*i] = self.f[3*i]
            self.indices[3*i+1] = self.f[3*i+1]
            self.indices[3*i+2] = self.f[3*i+2]

        for i in range(self.en):
            self.element[i] = [self.e[4*i], self.e[4*i+1], self.e[4*i+2], self.e[4*i+3]]
            #print(i, self.element[i])


    @ti.kernel
    def initialize(self):
        #总共有多少个tetrahedron
        for i in range(self.en):
            D = self.D(i)
            self.B[i] = D.inverse()
            a, b, c, d = self.element[i][0], self.element[i][1], self.element[i][2], self.element[i][3]
            self.element_volume[i] = abs(D.determinant()) / 6
            self.element_mass[i] = self.element_volume[i]
            self.node_mass[a] += self.element_mass[i]
            self.node_mass[b] += self.element_mass[i]
            self.node_mass[c] += self.element_mass[i]
            self.node_mass[d] += self.element_mass[i]
            self.neighbor_element_count[a] += 1
            self.neighbor_element_count[b] += 1
            self.neighbor_element_count[c] += 1
            self.neighbor_element_count[d] += 1

        for i in range(self.vn):
            self.node_mass[i] /= max(self.neighbor_element_count[i], 1)

    @ti.kernel
    def projection(self):
        for i in range(self.vn):
            self.position[i] = 0.2 * self.node[i]



    @ti.func
    def D(self, i):
        a = self.element[i][0]
        b = self.element[i][1]
        c = self.element[i][2]
        d = self.element[i][3]
        return ti.Matrix.cols([self.node[b]-self.node[a], self.node[c]-self.node[a], self.node[d]-self.node[a]])

    @ti.func
    def F(self, i): # deformation gradient
        return self.D(i) @ self.B[i]

    @ti.func
    def Psi(self, i): # (strain) energy density
        F = self.F(i)
        J = max(F.determinant(), 0.01)
        return self.mu / 2 * ( (F @ F.transpose()).trace() - self.dim ) - self.mu * ti.log(J) + self.la / 2 * ti.log(J)**2

    @ti.func
    def PK1(self, i):
        F = self.F(i)
        F_inv_T = F.inverse().transpose()
        J = max(F.determinant(), 0.01)
        return self.mu * (F - F_inv_T) + self.la * ti.log(J) * F_inv_T

    @ti.func
    def U0(self, i): # elastic potential energy
        return self.element_volume[i] * self.Psi(i)

    @ti.func
    def U1(self, i): # gravitational potential energy
        a = self.element[i][0]
        b = self.element[i][1]
        c = self.element[i][2]
        d = self.element[i][3]
        return self.element_mass[i] * 10 * 4 * (self.node[a].y + self.node[b].y + self.node[c].y + self.node[d].y) / 4


    @ti.kernel
    def energy_integrate(self):
        for i in range(self.en):
            self.energy[None] += self.U0(i) + self.U1(i)

    @ti.kernel
    def time_integrate(self, floor_height:ti.f32):
        
        mx = -self.inf

        for i in range(self.vn):
            self.velocity[i] += ( - self.node.grad[i] / self.node_mass[i]  ) * self.dt
            self.velocity[i] *= math.exp(self.dt*-6)
            self.node[i] += self.velocity[i] * self.dt

            if self.node[i].y < floor_height:
                self.node[i].y = floor_height
                self.velocity[i].y = 0

            mx = max(mx, self.node.grad[i].norm())

        mx = max(mx, 1)





#draw coordinate system as assitance
xi,yi,zi=20,20,20
position_xyz = ti.Vector.field(3,dtype=ti.f32,shape=xi+yi+zi)

@ti.kernel
def initialize_xyz():

    for i in range(xi):
        position_xyz[i] = ti.Vector([i/xi,0,0])
    for j in range(yi):
        position_xyz[xi+j]=ti.Vector([0,j/yi,0])
    for k in range(zi):
        position_xyz[xi+yi+k]=ti.Vector([0,0,k/zi])

initialize_xyz()


floor = Floor(0,1)
obj = Object('tetrahedral-models/ellell.1', 0)
obj.initialize()
floor.initialize()

#initialize GGUI render and camera
res = (1080, 1080)
lightpos = (0.5,0.5,0.5)
lightcolor = (1.0,1.0,1.0)
floorcolor = (0.5, 0.5, 0.5)
objectcolor = (0.5, 0.0, 0.0)

window = ti.ui.Window("Cloth",res,vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(1.0, 2.0, 2.0)
camera.lookat(0.0, 0.0, 0.0)
camera.fov(65)
ball_radius = 0.02

def render():
    #arcball in camera like FPS
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.2,0.0,0.0))
    scene.point_light(pos=lightpos, color=lightcolor)
    #rendering mesh and points
    scene.mesh(floor.position,
               floor.indices,
               color=floorcolor,
               per_vertex_color=None,
               two_sided=True)
    obj.projection()
    scene.mesh(obj.position,
               obj.indices,
               color=objectcolor,
               per_vertex_color=None,
               two_sided=True)
    scene.particles(position_xyz, 
                    radius=0.005, 
                    color=(0.0,1.0,1.0))


    #render scene
    canvas.scene(scene)


while window.running:
    for i in range(100):
        with ti.Tape(obj.energy):
            obj.energy_integrate()
        obj.time_integrate(floor.height)

    render()

    window.show()