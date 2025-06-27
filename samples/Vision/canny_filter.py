# -*- coding: utf-8 -*-
# @file canny_filter.py
# @brief The Canny Filter Example
# @author sailing-innocent
# @date 2025-03-11
# @version 1.0
# ---------------------------------

import argparse
import cv2
import torch
import kornia
import matplotlib.pyplot as plt

def apply_canny_filter(image_path):
    # Read the image using Matplotlib
    image = plt.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Convert the image to a tensor
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    print(image_tensor.shape) # torch.Size([1, 1, 512, 512])
    
    # Apply the Canny filter
    edges, _ = kornia.filters.Canny()(image_tensor)
    print(edges.shape) # torch.Size([1, 1, 512, 512])
    
    # Convert the tensor back to a numpy array
    edges_np = edges.squeeze().cpu().numpy() * 255.0
    
    # Display the original and edge-detected images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Canny Edges")
    plt.imshow(edges_np, cmap='gray')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Canny filter to an image using Kornia")
    parser.add_argument("--image_path", "-i", type=str, help="Path to the input image")
    args = parser.parse_args()
    
    apply_canny_filter(args.image_path)