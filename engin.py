# %%
import numpy as np
from functools import cached_property
import fresnel
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from vidgear.gears import WriteGear
import PIL
from io import BytesIO
from logger import get_logger
import logging
from multiprocessing import Pool
from collections import namedtuple

logger = get_logger('main', logging.INFO)


NodeState = namedtuple('NodeState',
                       field_names=[
                           'force',
                           'position',
                           'speed',
                           'mask',
                           'collide'
                       ]
                       )


class Nodes:
    def __init__(self, N=3) -> None:
        self.N = N  # maximum simulated objects
        self.pos = np.zeros((N, 3))
        self.speed = np.zeros((N, 3))
        self.mess = np.zeros((N, 1))
        self.mask = np.ones((N, 1))
        self.decay_accumulate=0

    @property
    def valid_map(self):
        return np.abs(self.mess) < 0.0000001

    @property
    def first_slot(self):
        nn, _ = np.where(self.valid_map)
        return nn[0]

    @cached_property
    def G(self):
        return 300

    @cached_property
    def r(self):
        return 2

    def add_node(self, position, speed, mass=0):
        slot = self.first_slot
        self.pos[slot] = position
        self.speed[slot] = speed
        self.mess[slot] = mass

    @staticmethod
    def _diff(pos) -> np.ndarray:
        p1 = pos.reshape(pos.shape[0], 1, pos.shape[1])
        p2 = pos.reshape(1, pos.shape[0], pos.shape[1])
        d = p1-p2
        return d

    @cached_property
    def speed_decay(self):
        return 0.9

    @property
    def pos_d(self):
        return self._diff(self.pos)

    @cached_property
    def max_force(self):
        return 100

    @staticmethod
    def _g_force(mess, pos, G, max_force):
        mm = mess.T * mess
        mm = mm.reshape((mm.shape[0], mm.shape[0], 1))

        p1 = pos.reshape(pos.shape[0], 1, pos.shape[1])
        p2 = pos.reshape(1, pos.shape[0], pos.shape[1])
        d = p1-p2

        # d = Nodes._diff(pos)
        pos_diff = d
        pos_diff_Ousq = pos_diff[:, :, 0]**2 + \
            pos_diff[:, :, 1] ** 2 + pos_diff[:, :, 2] ** 2
        pos_diff_Ousq = pos_diff_Ousq.reshape((
            pos_diff_Ousq.shape[0],
            pos_diff_Ousq.shape[1],
            1
        ))
        pos_diff_Ou = np.sqrt(pos_diff_Ousq)
        pos_direction = pos_diff / pos_diff_Ou
        F = G * mm * pos_direction * (1/pos_diff_Ou) * (1/pos_diff_Ou)
        F = F.clip(-max_force, max_force)
        F = np.nansum(F, axis=0)
        return F

    @property
    def g_force(self):
        """
            F = G*MM/(RELU(R-a) + a)
        """
        # return self._g_force(self.mess, self.pos, self.G, self.max_force)
        mm = self.mess.T * self.mess
        mm = mm.reshape(list(mm.shape) + [1])
        d = self.pos_d
        pos_diff = d
        pos_diff_Ousq = pos_diff[:, :, 0]**2 + \
            pos_diff[:, :, 1] ** 2 + pos_diff[:, :, 2] ** 2
        pos_diff_Ousq = pos_diff_Ousq[:, :, np.newaxis]
        pos_diff_Ou = np.sqrt(pos_diff_Ousq)
        pos_direction = pos_diff / pos_diff_Ou
        F = self.G * mm * pos_direction * (1/pos_diff_Ou) * (1/pos_diff_Ou)
        F = F.clip(-self.max_force, self.max_force)
        F = np.nansum(F, axis=0)
        return F

    def step(self, delta_t, with_state=False):
        self.pos += delta_t * self.speed * self.mask
        logger.debug(self.speed)
        logger.debug(self.mask)
        logger.debug(self.pos)
        pos_diff = self._diff(delta_t * self.speed + self.pos)
        pos_diff_Ousq = pos_diff[:, :, 0]**2 + \
            pos_diff[:, :, 1] ** 2 + pos_diff[:, :, 2] ** 2
        pos_diff_Ou = np.sqrt(pos_diff_Ousq)
        np.fill_diagonal(pos_diff_Ou, self.r+1)
        collide = pos_diff_Ou > self.r
        collide = collide.min(axis=1, keepdims=True)
        # logger.info(collide)
        g_force = self.g_force
        self.speed += g_force / self.mess * delta_t * self.mask
        self.decay_accumulate += delta_t
        if self.decay_accumulate > 0.1:
            self.decay_accumulate = 0
            self.speed = self.speed * self.speed_decay
        if with_state:
            return NodeState(
                g_force, 
                self.pos.copy(), 
                self.speed.copy(), 
                self.mask.copy(),
                collide,
                )


class Render:
    def __init__(self, node: Nodes) -> None:
        self.scene = fresnel.Scene()
        self.geometry = fresnel.geometry.Sphere(self.scene, N=node.N, radius=1)
        self.scene.camera = fresnel.camera.Orthographic(
            position=(0, 0, 400), look_at=(0, 0, 0), up=(0, 10, 0), height=160)

    def render(self, nodes: NodeState):
        self.geometry.position[:] = nodes.position
        output = fresnel.preview(self.scene)
        output = np.array(output.buf)
        output = output[:, :, :3]
        return output


class RenderCv:
    def __init__(self, lim=50, **kwargs) -> None:
        self.plt_args = kwargs
        self.lim = lim
        pass

    def render(self, nodes: NodeState):
        pos = nodes.position
        # logger.info(pos)
        # raise Exception()
        selection = (nodes.mask[:, 0] == 1)
        pos = pos[selection, :2]
        collide = nodes.collide[selection, 0]
        plt.scatter(
            pos[:, 0], pos[:, 1],
            c=collide.astype(np.int64),
            **self.plt_args)
        plt.colorbar()
        plt.xlim((-self.lim, self.lim))
        plt.ylim((-self.lim, self.lim))
        of = BytesIO()
        plt.savefig(of, format='png')
        of.seek(0)
        plt.close()
        img = cv2.imdecode(np.frombuffer(
            of.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        return img


class Simulator(object):
    def __init__(self, nodes: Nodes = None) -> None:
        self.nodes: Nodes = nodes
        self.render = RenderCv(s=1)
        # self.render = Render(nodes)

    def render_frame(self, state: NodeState):
        output = self.render.render(state)
        return output

    def on_step(self):
        pass

    def run(self, output_name: str, time=10.) -> None:
        node = self.nodes
        sims_per_frame = 20
        frame_rate = 30
        output_params = {"-vcodec": "libx264", "-crf": 0,
                         "-preset": "fast", "-input_framerate": frame_rate}
        step = 1/sims_per_frame/frame_rate
        pos_rec = []
        for i in tqdm(range(int(time)*frame_rate)):
            for i in range(sims_per_frame):
                state = node.step(step, with_state=(i == sims_per_frame-1))
            pos_rec.append(state)
        print('output...')
        if not output_name.endswith('.mp4'):
            output_name = output_name + '.mp4'
        result = WriteGear(
            output_filename=output_name, logging=False, **output_params)
        p = Pool()
        for output in p.map(self.render_frame, pos_rec):
            result.write(output)
        result.close()


# - %%

# %%
