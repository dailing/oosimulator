from engin import *

node = Nodes(3)
node.mess[:] = 4
node.pos[:] = [[0, -5, 0], [0, 5, 0], [5, 5, 0]]
node.speed[:] = [[7, 0, 0], [-7, 0, 0],  [0, 0, 0]]

print(__file__)

simulator = Simulator(node)
simulator.run('test2')

