                                        
#Problem1: rendering failure: count inside face, if open only ambient color, then everything will be fine, need to cull inside faces


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
        self.color = (0.95,0.99,0.97)


    @ti.kernel
    def initialize(self):
        h = self.height
        k = self.scale
        self.position[0] = [-k, h, -k]
        self.position[1] = [k, h, -k]
        self.position[2] = [k, h, k]
        self.position[3] = [-k, h, k]

        #floor indices
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

        self.velocity = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)
        self.force = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.vn)
        self.element_volume = ti.field(dtype=ti.f32, shape=self.en)
        self.B = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.en) # a square matrix

        print("vertices: ", self.vn, "elements: ", self.en)

        #view node as initial position, height=y+8.0
        for i in range(self.vn):
            self.node[i] = [self.v[3*i], self.v[3*i+1]+8.0, self.v[3*i+2]] 
            self.initial_node[i] = [self.v[3*i], self.v[3*i+1]+8.0, self.v[3*i+2]]
            self.velocity[i] = [0.0, 0.0, 0.0]

        for i in range(self.en):
            self.element[i] = [self.e[4*i], self.e[4*i+1], self.e[4*i+2], self.e[4*i+3]]

        #To be Continued: only need boundary faces, 
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
            self.velocity[i] = [0.0, 0.0, 0.0]

        for i in range(self.en):
            D = self.compute_D(i)
            # B is const matrix
            self.B[i] = D.inverse()
            #tetrahedron volume
            self.element_volume[i] = abs(D.determinant()) / 6.0

    #project node to position for rendering
    @ti.kernel
    def projection(self):
        for i in range(self.vn):
            self.position[i] = 0.2 * self.node[i]

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

    #compute the total energy
    @ti.kernel
    def energy(self) -> ti.f32:
        e = 0.0
        for i in range(self.en):
            e += self.element_volume[i] * self.Psi(i)
        for i in range(self.vn):
            e += self.node_mass * self.gravity * (self.node[i].y + 2)
            e += 0.5 * self.node_mass * self.velocity[i].dot(self.velocity[i])
        return e


    #compute force for each vertex E_total = E_strain + E_kinetic
    @ti.kernel
    def compute_force(self, node_mass:float, gravity:float):

        #add gravity
        for i in range(self.vn):
            self.force[i] = ti.Vector([0, -node_mass*gravity, 0])

        #add elasticity
        for i in range(self.en):
            #element_force = Wi * PK1 * partial(F)/partial(x)
            #partial(F)/partial(x) = ej * ei^T * B
            #PK1 * partial(F)/partial(x) = [Pk1@B^T]_kj 1*1 element
            #Hkj k-dim j-index
            H = - self.element_volume[i] * (self.PK1(i) @ self.B[i].transpose())

            #3-dim force 
            h1 = ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
            h2 = ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
            h3 = ti.Vector([H[0, 2], H[1, 2], H[2, 2]])

            a = self.element[i][0]
            b = self.element[i][1]
            c = self.element[i][2]
            d = self.element[i][3]

            self.force[a] += h1
            self.force[b] += h2
            self.force[c] += h3
            #inner force should be 0
            self.force[d] += -(h1 + h2 + h3)

    #Explicit Euler motion equation
    @ti.kernel
    def time_integrate(self, floor_height:ti.f32):

        mx = -self.inf

        #compute velocity and position of each point
        for i in range(self.vn):
            self.velocity[i] += ( self.force[i] / self.node_mass ) * self.dt
            self.node[i] += self.velocity[i] * self.dt

            #boundary conditions
            if self.node[i].y < floor_height:
                self.node[i].y = floor_height
                self.velocity[i].y = 0

            #ensure force not too large
            mx = max(mx, self.force[i].norm())

        mx = max(mx, 1)


#draw coordinate system as assitance
xi,yi,zi=20,20,20
position_xyz = ti.Vector.field(3,dtype=ti.f32,shape=xi+yi+zi)
@ti.kernel
def initialize_coordsys():
    for i in range(xi):
        position_xyz[i] = ti.Vector([i/xi,0,0])
    for j in range(yi):
        position_xyz[xi+j]=ti.Vector([0,j/yi,0])
    for k in range(zi):
        position_xyz[xi+yi+k]=ti.Vector([0,0,k/zi])
initialize_coordsys()


floor = Floor(0, 1)
obj = Object('tetrahedral-models/ellell.1', 0)

#object parameters
obj.node_mass = 1.0
obj.gravity = 10.0
obj.nu = 0.3
obj.E = 10000
obj.dt = 1e-4
obj.initialize()
floor.initialize()

#initialize GGUI render and camera
res = (1080, 1080)
lightpos = (0.0,1.0,1.0)
lightcolor = (1.0,1.0,1.0)
#lightcolor = (0.0,0.0,0.0)
floor.color = (0.95,0.99,0.97)
obj.color = (0.99,0.75,0.89)
ambientcolor = (0.0,0.0,0.0)

window = ti.ui.Window("Explicit FEM soft body",res,vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(1.0, 2.0, 2.0)
camera.lookat(0.0, 0.0, 0.0)
camera.fov(70)

def render():

    #arcball in camera like FPS, RMB rotate angle
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light(ambientcolor)
    scene.point_light(pos=lightpos, color=lightcolor)

    #rendering mesh and points
    scene.mesh(floor.position,
               floor.indices,
               color=floor.color,
               per_vertex_color=None,
               two_sided=True)

    #rendering object
    obj.projection()
    scene.mesh(obj.position,
               obj.indices,
               color=obj.color,
               per_vertex_color=None,
               two_sided=True)

    #rendering coordinate system
    scene.particles(position_xyz, 
                    radius=0.005, 
                    color=(0.0,1.0,1.0))

    #render scene
    canvas.scene(scene)

def imgui_options():
    window.GUI.begin("Presents", 0.05, 0.05, 0.2, 0.4)

    #parameters
    obj.E = window.GUI.slider_float("Elasticity modulus", obj.E, 5000, 20000)
    obj.nu = window.GUI.slider_float("Poisson ratio", obj.nu, 0.01, 0.49)
    obj.gravity = window.GUI.slider_float("gravity", obj.gravity, -10.0, 10.0)
    obj.node_mass = window.GUI.slider_float("node_mass", obj.node_mass, 0.0, 2.0)

    #control object floor light color
    obj.color = window.GUI.color_edit_3("objectcolor",obj.color)
    floor.color = window.GUI.color_edit_3("floorcolor",floor.color)

    if window.GUI.button("restart"):
        obj.initialize()


    window.GUI.end()


while window.running:

    #lame parameters
    obj.mu[None] = obj.E / (2 * (1 + obj.nu))
    obj.la[None] = obj.E * obj.nu / ((1 + obj.nu) * (1 - 2 * obj.nu))

    for i in range(100):
        obj.compute_force(obj.node_mass, obj.gravity)
        obj.time_integrate(floor.height)

    render()
    imgui_options()

    window.show()