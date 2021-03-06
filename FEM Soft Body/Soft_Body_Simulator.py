
import taichi as ti #version 0.8.7
import taichi_glsl as ts


import sys
sys.path.append('FEM Soft Body')
from Implicit_FEM_Soft_Body import *
#from test_new_implicit import *


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
class Soft_Body_Simulator:
    def __init__(self):
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.xi, self.yi, self.zi = 20, 20, 20
        self.position_xyz = ti.Vector.field(3,dtype=ti.f32,shape=self.xi+self.yi+self.zi)


    #draw coordinate system as assitance
    @ti.kernel
    def initialize_coordsys(self):
        for i in range(self.xi):
            self.position_xyz[i] = ti.Vector([i/self.xi,0,0])
        for j in range(self.yi):
            self.position_xyz[self.xi+j]=ti.Vector([0,j/self.yi,0])
        for k in range(self.zi):
            self.position_xyz[self.xi+self.yi+k]=ti.Vector([0,0,k/self.zi])


@ti.kernel
def implicit_time_integrate(floor_height:ti.f32, modelscale:ti.f32):

    #compute velocity and position of each point
    for i in range(obj.vn):
        for j in ti.static(range(obj.dim)):
            obj.velocity[i*obj.dim+j] = (obj.node[i][j] - obj.previous_node[i][j]) / obj.dt

        #boundary conditions
        if obj.node[i].y < floor_height:
            obj.node[i].y = floor_height
            obj.velocity[obj.dim*i+1] = 0.0

    # object position projection
    for i in range(obj.vn):
        obj.position[i] = modelscale * obj.node[i]



floor = Floor(0, 1)
obj = Implicit_Object('tetrahedral-models/ellell.1', 0)
simulator = Soft_Body_Simulator()


#object parameters
obj.initialize()
floor.initialize()
simulator.initialize_coordsys()


#initialize GGUI render and camera
res = (1080, 1080)
lightpos = (0.0,1.0,1.0)
lightcolor = (1.0,1.0,1.0)
floor.color = (0.95,0.99,0.97)
obj.color = (0.99,0.75,0.89)
ambientcolor = (0.0,0.0,0.0)

window = ti.ui.Window("Semi-Implicit FEM soft body",res,vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(1.0, 2.0, 2.0)
camera.lookat(0.0, 0.0, 0.0)
camera.fov(70)

#linear equation solver
linear_equation_solver = [
    "Jacobi iteration",
    "Conjugate Gradient",
    "ICC Precondition CG",
    ]
implicit_method = [
    "Semi-implicit",
    "Implicit",
    ]

curr_equation_solver = 0
curr_implicit_method = 0

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
    scene.mesh(obj.position,
               obj.indices,
               color=obj.color,
               per_vertex_color=None,
               two_sided=True)

    #rendering coordinate system
    scene.particles(simulator.position_xyz, 
                    radius=0.005, 
                    color=(0.0,1.0,1.0))

    #render scene
    canvas.scene(scene)


def imgui_options():
    window.GUI.begin("Presents", 0.05, 0.05, 0.2, 0.4)

    #parameters
    obj.E = window.GUI.slider_float("Elasticity modulus", obj.E, 1000, 20000)
    obj.nu = window.GUI.slider_float("Poisson ratio", obj.nu, 0.01, 0.49)
    obj.gravity = window.GUI.slider_float("gravity", obj.gravity, -10.0, 10.0)
    obj.node_mass = window.GUI.slider_float("node_mass", obj.node_mass, 0.0, 2.0)

    #control object floor light color
    obj.color = window.GUI.color_edit_3("objectcolor",obj.color)
    floor.color = window.GUI.color_edit_3("floorcolor",floor.color)
    obj.modelscale = window.GUI.slider_float("modelscale", obj.modelscale, 0.1, 0.4)

    global curr_equation_solver
    old_equation_solver = curr_equation_solver
    for i in range(len(linear_equation_solver)):
        if window.GUI.checkbox(linear_equation_solver[i], curr_equation_solver == i):
            curr_equation_solver = i
    if curr_equation_solver != old_equation_solver:
        obj.initialize()

    global curr_implicit_method
    old_implicit_method = curr_implicit_method
    for i in range(len(implicit_method)):
        if window.GUI.checkbox(implicit_method[i], curr_implicit_method == i):
            curr_implicit_method = i
    if curr_implicit_method != old_implicit_method:
        obj.initialize()

    if window.GUI.button("restart"):
        obj.initialize()

    window.GUI.end()


while window.running:

    #lame parameters
    obj.mu[None] = obj.E / (2 * (1 + obj.nu))
    obj.la[None] = obj.E * obj.nu / ((1 + obj.nu) * (1 - 2 * obj.nu))

    # in dt=1e-2 range=1
    # in dt=1e-3 range=10
    for i in range(10): 
        
        if curr_implicit_method == 0:        
            obj.one_step_Newton()

        elif curr_implicit_method == 1:
            #obj.ordinary_Newton(100, 1e-6)
            obj.damped_Newton(100, 1e-6)

        implicit_time_integrate(floor.height, obj.modelscale)

    render()
    imgui_options()
    
    window.show()

