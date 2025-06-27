# -*- coding: utf-8 -*-
# @file marcher.py
# @brief The Ray Marcher Implementation
# @author sailing-innocent
# @date 2025-02-24
# @version 1.0
# ---------------------------------

import numpy as np 
from utils import ray_sphere_intersect
from utils import Sphere 
import matplotlib.pyplot as plt

def integrate(ray_origin, ray_direction, obj, bg_color = np.array([0.572, 0.772, 0.921])):
    flag, inside, isect = ray_sphere_intersect(ray_origin, ray_direction, obj)
    color = bg_color
    # ray marcher
    if flag:
        step_size = 0.2
        absorption = 0.1
        scatting = 0.1
        density = 1.0
        ns = np.ceil((isect.t1 - isect.t0) / step_size)
        light_dir = np.array([0.0, 1.0, 0.0])
        light_color = np.array([1.3, 0.3, 0.9])
        transparency = 1.0
        result = np.array([0.0, 0.0, 0.0])
        for i in range(int(ns)):
            # FROM BACK TO FRONT
            t = isect.t1 - (i + 0.5) * step_size
            p = ray_origin + t * ray_direction
            sample_transparency = np.exp(-step_size * (scatting + absorption))
            transparency *= sample_transparency
            flag, vinside, vsect = ray_sphere_intersect(p, light_dir, obj)
            if flag and vinside:
                light_attenuation = np.exp(-density * vsect.t1 * (scatting + absorption))
                result += light_attenuation * light_color * scatting * density * step_size
            
            result *= sample_transparency
        color = bg_color * transparency + result
    return color

if __name__ == "__main__":
    W = 256
    H = 128
    i, j = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    aspect = W / H
    dirs = np.stack([(i-0.5)/1.0 * aspect, -(j-0.5)/1, -np.ones_like(i)], -1)
    assert dirs.shape == (H, W, 3)
    center = np.array([0.0, 0.0, 0.0])
    radius = 5.0
    obj = Sphere(center, radius)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    pixels = np.ones_like(dirs)

    for i in range(W):
        for j in range(H):
            ro = np.array([0,0,20])
            rd = dirs[j,i,:]
            pixels[j,i,:] = integrate(ro, rd, obj)
    
    plt.imshow(pixels)
    plt.show()