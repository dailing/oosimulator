# %%
import pickle
from engin import *
import numpy as np

fixed_nodes = np.array([[1, 10, 0], [-1, 10, 0], [0, -10, 0]])
print(fixed_nodes.shape)
print(len(fixed_nodes))


node = Nodes(160)

node.pos[-len(fixed_nodes):] = fixed_nodes
node.mess[-len(fixed_nodes):] = 4
node.mask[-len(fixed_nodes):] = 0.0


N = 150
pos = []
while len(pos) < N:
    trail = np.random.rand(1, 3)*60 - 30
    distance = np.sum((trail - fixed_nodes)**2, axis=1)
    pos.append(trail[0])
pos = np.array(pos)

node.mess[:N] = 0.3
node.pos[:N] = pos
node.speed[:N] = np.random.rand(N, 3) * 20 - 10

simulator = Simulator(node)
simulator.run('test4', 30)

# def trail_once(node: Nodes):
#     pass


# trail_node = Nodes(node.N)

# trail_node.pos[:] = node.pos
# trail_node.mess[:] = node.mess
# trail_node.mask[:] = node.mask

# # %%

# sim_time = 20.0
# step_size = 0.05
# for _ in range(int(sim_time/step_size)):
#     trail_node.step(step_size)
# plt.scatter(trail_node.pos[:,0], trail_node.pos[:, 1], s=0.2)
# plt.xlim(-20, 20)
# plt.ylim(-20, 20)
# plt.show()
# print(np.nansum(trail_node.speed ** 2))

# %%

# %%
