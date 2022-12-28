import pickle
from engin import *

fixed_nodes = pickle.load(open('wcenter.pkl', 'rb')) * 20
print(fixed_nodes.shape)
print(len(fixed_nodes))


node = Nodes(200)

node.pos[-len(fixed_nodes):] = fixed_nodes
node.mess[-len(fixed_nodes):] = 0.3
node.mask[-len(fixed_nodes):] = 0.0


node.mess[:3] = 4
node.pos[:3] = [[0, -5, 0], [0, 5, 0], [5, 5, 3]]
node.speed[:3] = [[7, 0, 0], [-7, 0, 0],  [0, 0, 3]]


simulator = Simulator(node)
simulator.run('test3')

