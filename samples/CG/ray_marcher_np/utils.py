# -*- coding: utf-8 -*-
# @file utils.py
# @brief Ray Marcher Utils
# @author sailing-innocent
# @date 2025-02-24
# @version 1.0
# ---------------------------------

import numpy as np 
import unittest 

class ISectData:
    t0: float = 0.0
    t1: float = 1.0
    pHit: np.ndarray = np.array([0,0,0])
    nHit: np.ndarray = np.array([0,0,0])

    def __init__(self, t0 = 0.0, t1 = 1.0, pHit = np.array([0,0,0]), nHit = np.array([0,0,0])):
        self.t0 = t0
        self.t1 = t1
        self.pHit = pHit
        self.nHit = nHit

class Sphere:
    center: np.ndarray = np.array([0,0,0])
    radius: float = 1.0
    def __init__(self, center = np.array([0.0,0.0,0.0]), radius = 1.0):
        self.center = center
        self.radius = radius

def solve_quadratic(a, b, c):
    d = b * b - 4 * a * c
    flag = False
    x1 = 0.0
    x2 = 0.0
    if d < 0:
        return flag, x1, x2
    elif d > 0:
        d = np.sqrt(d)
        x1 = (-b - d) / (2 * a)
        x2 = (-b + d) / (2 * a)
        flag = True
    else:
        x1 = x2 = -b / (2 * a)
        flag = True
    
    return flag, x1, x2

def ray_sphere_intersect(ro, rd, s):
    a = np.dot(rd, rd)
    b = 2 * np.dot(rd, ro - s.center)
    c = np.dot(ro - s.center, ro - s.center) - s.radius * s.radius
    flag, x1, x2 = solve_quadratic(a, b, c)
    inside = False
    isect = ISectData(x1, x2)
    if (flag):
        if (isect.t0 < 0):
            if (isect.t1 < 0):
                flag = False
            else:
                inside = True
                isect.t0 = 0
    return flag, inside, isect


# ---------------------------------
# Test Cases
# ---------------------------------
class TestRayMarchUtil(unittest.TestCase):
    def test_quadratic_01(self):
        # a^2 - 3a + 2 = 0
        a = 1
        b = -3
        c = 2
        flag, x1, x2 = solve_quadratic(a, b, c)
        assert flag
        assert x1 == 1
        assert x2 == 2


    def test_quadratic_02(self):
        # 2a^2 - 8a + 6 = 0
        a = 2
        b = -8
        c = 6
        flag, x1, x2 = solve_quadratic(a, b, c)
        assert flag
        assert x1 == 1.0
        assert x2 == 3.0

    def test_quadratic_03(self):
        # a^2 + 2a + 1 = 0
        a = 1
        b = 2
        c = 1
        flag, x1, x2 = solve_quadratic(a, b, c)
        assert flag
        assert x1 == -1
        assert x2 == -1

    def test_quadratic_04(self):
        # a^2 + 2a + 2 = 0
        a = 1
        b = 2
        c = 2
        flag, x1, x2 = solve_quadratic(a, b, c)
        assert not flag

    def test_ray_sphere_intersect(self):
        sphere = Sphere()
        ray_origin = np.array([0, 0, 3])
        ray_direction = np.array([0, 0, -1])
        flag, inside, isect = ray_sphere_intersect(ray_origin, ray_direction, sphere)
        assert flag
        assert not inside
        assert isect.t0 == 2.0
        assert isect.t1 == 4.0

if __name__ == '__main__':
    unittest.main()