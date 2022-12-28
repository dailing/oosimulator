# %%
# x^2+(y-x^2_3)^2=1
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pickle

# %%
xx,yy = np.meshgrid(np.linspace(2, -2, 50), np.linspace(2, -2, 50))
# %%
error = np.abs(xx**2 + (yy-(xx**2)**(1/3))**2 -1) < 0.09

plt.imshow(error)
plt.colorbar()
plt.show()
# %%
inx_x, idx_y = np.where(error > 0)
# %%
px = xx[inx_x, idx_y]
py = yy[inx_x, idx_y]
# %%
plt.scatter(px, py)
plt.show()

# %%
pos_centers = np.stack([px, py, np.zeros_like(px)], axis=-1)
# %%
pickle.dump(pos_centers, open('wcenter.pkl', 'wb'))

# %%
