# %%
from engin import *
from PIL import Image

node = Nodes(3)
node.mess[:] = 4
node.pos[:] = [[0, 6, 0], [0, 5, 0], [0, 1, 0]]
# node.speed[:] = [[7, 0, 0], [7, 0, 0]]

print(__file__)
state = node.step(0.01, True)

simulator = Simulator(node)
output = simulator.render_frame(state)
Image.fromarray(output)
# simulator.run('test1', time=0.2)
# x^2+(y-x^2_3)^2=1
# %%
