import pickle
from engin import *
import numpy as np

fixed_nodes = pickle.load(open('wcenter.pkl', 'rb')) * 20
print(fixed_nodes.shape)
print(len(fixed_nodes))


node = Nodes(300)

node.pos[-len(fixed_nodes):] = fixed_nodes
node.mess[-len(fixed_nodes):] = 4
node.mask[-len(fixed_nodes):] = 0.0


N = 200
pos = []
while len(pos) < N:
    trail = np.random.rand(1, 3)*60 - 30
    distance = np.sum((trail - fixed_nodes)**2, axis=1)
    if distance.min() < 30:
        pos.append(trail[0])
pos = np.array(pos)

node.mess[:N] = 0.3
node.pos[:N] = pos
node.speed[:N] = np.random.rand(N, 3) * 20 - 10


simulator = Simulator(node)
simulator.run('test3', 20)
