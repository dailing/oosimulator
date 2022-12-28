# %%
"""
good test for 3d render for now

"""
import fresnel
import numpy
import PIL
import sys
import os

scene = fresnel.Scene()
geometry = fresnel.geometry.Sphere(scene, N=8, radius=1)
geometry.position[:] = [[1,1,1],
                        [1,1,-1],
                        [1,-1,1],
                        [1,-1,-1],
                        [-1,1,1],
                        [-1,1,-1],
                        [-1,-1,1],
                        [-1,-1,-1]]
# scene.camera = fresnel.camera.Orthographic.fit(scene)
scene.camera = fresnel.camera.Orthographic(
    position=(2,2,2), look_at=(0,0,0), up=(0,1,0), height=8)

# %\
fresnel.preview(scene)


# %%
fresnel.preview(scene, anti_alias=False)

# %%
output = fresnel.pathtrace(scene, light_samples=40, samples=128)
output
# %%
