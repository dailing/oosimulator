from engin import *

node = Nodes(2)
node.mess[:] = 4
node.pos[:] = [[0, -5, 0], [0, 5, 0]]
node.speed[:] = [[7, 0, 0], [-7, 0, 0]]

print(__file__)

simulator = Simulator(node)
simulator.run('test1')
# x^2+(y-x^2_3)^2=1