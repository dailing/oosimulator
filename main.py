# %%
import numpy as np
from functools import cached_property
import fresnel
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

class Nodes:
    def __init__(self, N=3) -> None:
        self.N = N  # maximum simulated objects
        self.pos = np.zeros((N, 3))
        self.speed = np.zeros((N, 3))
        self.mess = np.zeros((N, 1))

    @property
    def valid_map(self):
        return np.abs(self.mess) < 0.0000001

    @property
    def first_slot(self):
        nn, _ = np.where(self.valid_map)
        return nn[0]

    @cached_property
    def G(self):
        return 1

    def add_node(self, position, speed, mass=0):
        slot = self.first_slot
        self.pos[slot] = position
        self.speed[slot] = speed
        self.mess[slot] = mass

    @property
    def g_force(self):
        """
            F = G*MM/R^2
        """
        mm = self.mess.T * self.mess
        mm = mm.reshape(list(mm.shape) + [1])
        p1 = self.pos.reshape(self.pos.shape[0], 1, self.pos.shape[1])
        p2 = self.pos.reshape(1, self.pos.shape[0], self.pos.shape[1])
        d = p1-p2
        d_2 = d[:, :, 0]**2 + d[:, :, 1]**2
        d_2 = d_2.reshape(list(d_2.shape)+[1])
        d_sq_2 = np.sqrt(d_2)
        # print(mm.shape)
        F = self.G * mm * d * (1/d_2) * (1/d_sq_2)
        for i in range(3):
            np.fill_diagonal(F[:, :, i], 0)
        F = F.sum(axis=0)
        return F

    def step(self, delta_t):
        self.pos += delta_t * self.speed
        self.speed += self.g_force / self.mess * delta_t

        pos_diff = self.pos


class Render:
    def __init__(self) -> None:
        self.scene = fresnel.Scene()
        self.geometry = fresnel.geometry.Sphere(self.scene, N=2, radius=0.2)
        # geometry.position[:] = []
        # scene.camera = fresnel.camera.Orthographic.fit(scene)
        self.scene.camera = fresnel.camera.Orthographic(
            position=(0, 0, 20), look_at=(0, 0, 0), up=(0, 1, 0), height=8)

    def render(self, nodes: Nodes):
        self.geometry.position[:] = nodes.pos
        output = fresnel.preview(self.scene)
        output = np.array(output.buf)
        output = output[:, :, :3]
        return output


class Simulator(object):
    def __init__(self) -> None:
        self.nodes = None

    def add_obj(self, pos, speed=(0, 1, 0), mass=1, ):
        pass


# %%
node = Nodes(2)
node.mess[1] = 1
node.mess[0] = 1
node.pos[0] = (0, 1, 0)
node.pos[1] = (0, -1, 0)
node.speed[0] = (0, 0, 0)
node.speed[0] = (0, 0, 0)
render = Render()
result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (600,370))

for i in tqdm(range(120)):
    node.step(1/30)
    output = render.render(node)
    result.write(output)
result.release()



# %%
