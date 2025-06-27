# -*- coding: utf-8 -*-
# @file diff_gs_3d.py
# @brief 3D Gaussian 
# @author sailing-innocent
# @date 2025-02-24
# @version 1.0
# ---------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quaternion_to_matrix(q):
    """Convert quaternion to rotation matrix"""
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])

def get_covariance_matrix_3d(sigma, q):
    """Convert diagonal sigma and quaternion rotation to full 3D covariance matrix"""
    R = quaternion_to_matrix(q)
    D = np.diag(sigma**2)
    return R @ D @ R.T

def kl_divergence_3d(mu, sigma, q, target_mu, target_sigma, target_q):
    """
    KL divergence between 3D Gaussians with full covariance matrices
    """
    Sigma1 = get_covariance_matrix_3d(sigma, q)
    Sigma2 = get_covariance_matrix_3d(target_sigma, target_q)
    Sigma2_inv = np.linalg.inv(Sigma2)
    
    diff = mu - target_mu
    
    return 0.5 * (
        np.trace(Sigma2_inv @ Sigma1) +
        diff.T @ Sigma2_inv @ diff -
        3 + np.log(np.linalg.det(Sigma2)/np.linalg.det(Sigma1))
    )

def gradients_3d(mu, sigma, q, target_mu, target_sigma, target_q):
    """Compute analytical gradients for 3D Gaussian parameters including quaternion rotation"""
    # Get covariance matrices and their properties
    Sigma1 = get_covariance_matrix_3d(sigma, q)
    Sigma2 = get_covariance_matrix_3d(target_sigma, target_q)
    Sigma2_inv = np.linalg.inv(Sigma2)
    Sigma1_inv = np.linalg.inv(Sigma1)
    
    # Compute difference vector
    diff = mu - target_mu
    
    # Gradient for mu (straightforward from KL divergence formula)
    d_mu = Sigma2_inv @ diff
    
    # Gradient for sigma
    R = quaternion_to_matrix(q)
    d_sigma = np.zeros(3)
    for i in range(3):
        D_i = np.zeros((3, 3))
        D_i[i,i] = 2 * sigma[i]  # Derivative of D with respect to sigma[i]
        dSigma1_dsigma = R @ D_i @ R.T
        d_sigma[i] = 0.5 * (
            np.trace(Sigma2_inv @ dSigma1_dsigma) - 
            np.trace(Sigma1_inv @ dSigma1_dsigma)
        )
    
    # Gradient for quaternion
    d_q = np.zeros(4)
    w, x, y, z = q
    
    # Partial derivatives of rotation matrix with respect to quaternion components
    dR_dw = np.array([
        [0, -2*z, 2*y],
        [2*z, 0, -2*x],
        [-2*y, 2*x, 0]
    ])
    dR_dx = np.array([
        [0, 2*y, 2*z],
        [2*y, -4*x, 0],
        [2*z, 0, -4*x]
    ])
    dR_dy = np.array([
        [-4*y, 2*x, 0],
        [2*x, 0, 2*z],
        [0, 2*z, -4*y]
    ])
    dR_dz = np.array([
        [-4*z, 0, 2*x],
        [0, -4*z, 2*y],
        [2*x, 2*y, 0]
    ])
    
    D = np.diag(sigma**2)
    dR_list = [dR_dw, dR_dx, dR_dy, dR_dz]
    
    for i in range(4):
        dR = dR_list[i]
        dSigma1_dq = dR @ D @ R.T + R @ D @ dR.T
        d_q[i] = 0.5 * (
            np.trace(Sigma2_inv @ dSigma1_dq) - 
            np.trace(Sigma1_inv @ dSigma1_dq)
        )
    
    # Normalize quaternion gradient to maintain unit norm constraint
    q_norm = np.linalg.norm(q)
    d_q = d_q - q * (q @ d_q) / (q_norm * q_norm)
    
    return d_mu, d_sigma, d_q

# def gradients_3d(mu, sigma, q, target_mu, target_sigma, target_q):
#     """Compute numerical gradients for 3D Gaussian parameters including quaternion rotation"""
#     eps = 1e-6
    
#     # Numerical gradients
#     d_mu = np.zeros(3)
#     d_sigma = np.zeros(3)
#     d_q = np.zeros(4)
    
#     # Gradient for mu
#     for i in range(3):
#         mu_plus = mu.copy()
#         mu_plus[i] += eps
#         mu_minus = mu.copy()
#         mu_minus[i] -= eps
#         d_mu[i] = (kl_divergence_3d(mu_plus, sigma, q, target_mu, target_sigma, target_q) -
#                    kl_divergence_3d(mu_minus, sigma, q, target_mu, target_sigma, target_q)) / (2*eps)
    
#     # Gradient for sigma
#     for i in range(3):
#         sigma_plus = sigma.copy()
#         sigma_plus[i] += eps
#         sigma_minus = sigma.copy()
#         sigma_minus[i] -= eps
#         d_sigma[i] = (kl_divergence_3d(mu, sigma_plus, q, target_mu, target_sigma, target_q) -
#                      kl_divergence_3d(mu, sigma_minus, q, target_mu, target_sigma, target_q)) / (2*eps)
    
#     # Gradient for quaternion
#     for i in range(4):
#         q_plus = q.copy()
#         q_plus[i] += eps
#         q_plus = q_plus / np.linalg.norm(q_plus)
#         q_minus = q.copy()
#         q_minus[i] -= eps
#         q_minus = q_minus / np.linalg.norm(q_minus)
#         d_q[i] = (kl_divergence_3d(mu, sigma, q_plus, target_mu, target_sigma, target_q) -
#                   kl_divergence_3d(mu, sigma, q_minus, target_mu, target_sigma, target_q)) / (2*eps)
    
#     return d_mu, d_sigma, d_q

def gaussian_3d(X, Y, Z, mu, sigma, q):
    """3D Gaussian with quaternion rotation"""
    Sigma = get_covariance_matrix_3d(sigma, q)
    Sigma_inv = np.linalg.inv(Sigma)
    
    # Calculate coordinates relative to mean
    coords = np.stack([(X - mu[0]).flatten(),
                      (Y - mu[1]).flatten(),
                      (Z - mu[2]).flatten()], axis=1)  # Shape: (N, 3)
    
    # Calculate quadratic form for each point
    quad = np.sum(coords @ Sigma_inv * coords, axis=1)  # Shape: (N,)
    
    # Reshape back to original grid shape
    quad = quad.reshape(X.shape)
    
    # Calculate normalization constant
    norm_const = 1 / ((2 * np.pi) ** (3/2) * np.sqrt(np.linalg.det(Sigma)))
    
    return norm_const * np.exp(-0.5 * quad)

if __name__ == "__main__":
    # Target parameters
    target_mu = np.array([2.0, 1.0, 1.5])
    target_sigma = np.array([2.0, 1.5, 1.0])
    # Quaternion representing 45-degree rotation around z-axis
    target_q = np.array([np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)])
    target_q = target_q / np.linalg.norm(target_q)

    # Initialize parameters
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([1.0, 1.0, 1.0]) # scaling factor for different direction
    q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    learning_rate = 0.1
    num_iterations = 200

    loss_history = []
    mu_history = []
    sigma_history = []
    q_history = []

    for i in range(num_iterations):
        loss = kl_divergence_3d(mu, sigma, q, target_mu, target_sigma, target_q)
        loss_history.append(loss)
        mu_history.append(mu.copy())
        sigma_history.append(sigma.copy())
        q_history.append(q.copy())
        
        d_mu, d_sigma, d_q = gradients_3d(mu, sigma, q, target_mu, target_sigma, target_q)
        
        # Update parameters
        mu = mu - learning_rate * d_mu
        sigma = sigma - learning_rate * d_sigma
        q = q - learning_rate * d_q
        
        # Ensure sigma remains positive and quaternion normalized
        sigma = np.maximum(sigma, 1e-6)
        q = q / np.linalg.norm(q)
        
        if (i + 1) % 20 == 0:
            print(f"Iteration {i+1:03d}: Loss = {loss:.6f}")
            print(f"mu = [{mu[0]:.3f}, {mu[1]:.3f}, {mu[2]:.3f}]")
            print(f"sigma = [{sigma[0]:.3f}, {sigma[1]:.3f}, {sigma[2]:.3f}]")
            print(f"q = [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('KL Divergence')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.show()

    # 3D visualization of final state
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # z=0 plane

    # Calculate densities at z=0 plane
    density_target = gaussian_3d(X, Y, Z, target_mu, target_sigma, target_q)
    density_final = gaussian_3d(X, Y, Z, mu, sigma, q)

    # Plot contours
    fig = plt.figure(figsize=(12, 6))
    
    # Target distribution
    ax1 = fig.add_subplot(121)
    ax1.scatter(target_mu[0], target_mu[1], color='r', s=100, label='Mean')
    ax1.contour(X, Y, density_target, levels=10)
    ax1.set_title('Target Distribution (z=0 slice)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    
    # Final distribution
    ax2 = fig.add_subplot(122)
    ax2.scatter(mu[0], mu[1], color='r', s=100, label='Mean')
    ax2.contour(X, Y, density_final, levels=10)
    ax2.set_title('Optimized Distribution (z=0 slice)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Optional: Add 3D surface plot
    fig = plt.figure(figsize=(12, 6))
    
    # Target distribution
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, density_target, cmap='viridis', alpha=0.8)
    ax1.scatter(target_mu[0], target_mu[1], np.max(density_target), 
                color='r', s=100, label='Mean')
    ax1.set_title('Target Distribution (z=0 slice)')
    
    # Final distribution
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, density_final, cmap='viridis', alpha=0.8)
    ax2.scatter(mu[0], mu[1], np.max(density_final), 
                color='r', s=100, label='Mean')
    ax2.set_title('Optimized Distribution (z=0 slice)')

    plt.tight_layout()
    plt.show()