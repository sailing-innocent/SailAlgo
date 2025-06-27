# -*- coding: utf-8 -*-
# @file dummy.py
# @brief Dummy Diagram Generator, generate a placeholder pure white figure 
# @author sailing-innocent
# @date 2025-03-19
# @version 1.0
# ---------------------------------

import matplotlib.pyplot as plt
import numpy as np 

def draw(name: str, outdir: str):
    w = 1024
    h = 768
    fig = plt.figure(name, figsize=(w/100, h/100), dpi=100)
    fig.patch.set_facecolor('white')
    plt.gca().set_facecolor('white')
    plt.axis('off')
    plt.savefig(f"{outdir}/{name}.png")
    return True 


