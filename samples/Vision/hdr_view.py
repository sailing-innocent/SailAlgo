# -*- coding: utf-8 -*-
# @file hdr_view.py
# @brief Details of HDR file 
# @author sailing-innocent
# @date 2025-02-21
# @version 1.0
# ---------------------------------

import imageio.v3 as iio 
import argparse 
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View HDR file")
    parser.add_argument("--file", help="HDR file", default="data/assets/images/qwantani_noon_4k.exr")
    args = parser.parse_args()
    hdr = iio.imread(args.file)
    # print the basic info
    print(hdr.dtype)
    print(hdr.min()) # -0.005
    print(hdr.max()) # 137869
    print(hdr.mean()) # 3.12
    print(hdr.std()) # 209.8105
    print(hdr.shape) # (2048, 4096, 4)
