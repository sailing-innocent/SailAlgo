# -*- coding: utf-8 -*-
# @file diff_gs_2d.py
# @brief The Differential Gaussian 2D 
# @author sailing-innocent
# @date 2025-02-24
# @version 1.0
# ---------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotation_matrix(theta):
    """Create 2D rotation matrix"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def get_covariance_matrix(sigma, theta):
    """Convert diagonal sigma and rotation theta to full covariance matrix"""
    R = rotation_matrix(theta)
    D = np.diag(sigma**2)
    return R @ D @ R.T

def kl_divergence_2d(mu, sigma, theta, target_mu, target_sigma, target_theta):
    """
    KL divergence between 2D Gaussians with full covariance matrices
    """
    Sigma1 = get_covariance_matrix(sigma, theta)
    Sigma2 = get_covariance_matrix(target_sigma, target_theta)
    Sigma2_inv = np.linalg.inv(Sigma2)
    
    diff = mu - target_mu
    
    return 0.5 * (
        np.trace(Sigma2_inv @ Sigma1) +
        diff.T @ Sigma2_inv @ diff -
        2 + np.log(np.linalg.det(Sigma2)/np.linalg.det(Sigma1))
    )

def gradients_2d(mu, sigma, theta, target_mu, target_sigma, target_theta):
    """Compute analytical gradients for 2D Gaussian parameters including rotation"""
    # Get covariance matrices and their properties
    Sigma1 = get_covariance_matrix(sigma, theta)
    Sigma2 = get_covariance_matrix(target_sigma, target_theta)
    Sigma2_inv = np.linalg.inv(Sigma2)
    
    # Compute difference vector
    diff = mu - target_mu
    
    # Gradient for mu
    d_mu = Sigma2_inv @ diff
    
    # Gradient for sigma
    R = rotation_matrix(theta)
    d_sigma = np.zeros(2)
    for i in range(2):
        D_i = np.zeros((2, 2))
        D_i[i,i] = 2 * sigma[i]  # Derivative of D with respect to sigma[i]
        dSigma1_dsigma = R @ D_i @ R.T
        d_sigma[i] = 0.5 * (
            np.trace(Sigma2_inv @ dSigma1_dsigma) - 
            np.trace(np.linalg.inv(Sigma1) @ dSigma1_dsigma)
        )
    
    # Gradient for theta
    dR = np.array([[-np.sin(theta), -np.cos(theta)],
                   [np.cos(theta), -np.sin(theta)]])
    D = np.diag(sigma**2)
    dSigma1_dtheta = dR @ D @ R.T + R @ D @ dR.T
    
    d_theta = 0.5 * (
        np.trace(Sigma2_inv @ dSigma1_dtheta) - 
        np.trace(np.linalg.inv(Sigma1) @ dSigma1_dtheta)
    )
    
    return d_mu, d_sigma, d_theta


# def gradients_2d(mu, sigma, theta, target_mu, target_sigma, target_theta):
#     """Compute gradients for 2D Gaussian parameters including rotation"""
#     eps = 1e-6
    
#     # Numerical gradients for now (for simplicity)
#     d_mu = np.zeros(2)
#     d_sigma = np.zeros(2)
    
#     # Gradient for mu
#     for i in range(2):
#         mu_plus = mu.copy()
#         mu_plus[i] += eps
#         mu_minus = mu.copy()
#         mu_minus[i] -= eps
#         d_mu[i] = (kl_divergence_2d(mu_plus, sigma, theta, target_mu, target_sigma, target_theta) -
#                    kl_divergence_2d(mu_minus, sigma, theta, target_mu, target_sigma, target_theta)) / (2*eps)
    
#     # Gradient for sigma
#     for i in range(2):
#         sigma_plus = sigma.copy()
#         sigma_plus[i] += eps
#         sigma_minus = sigma.copy()
#         sigma_minus[i] -= eps
#         d_sigma[i] = (kl_divergence_2d(mu, sigma_plus, theta, target_mu, target_sigma, target_theta) -
#                      kl_divergence_2d(mu, sigma_minus, theta, target_mu, target_sigma, target_theta)) / (2*eps)
    
#     # Gradient for theta
#     theta_plus = theta + eps
#     theta_minus = theta - eps
#     d_theta = (kl_divergence_2d(mu, sigma, theta_plus, target_mu, target_sigma, target_theta) -
#                kl_divergence_2d(mu, sigma, theta_minus, target_mu, target_sigma, target_theta)) / (2*eps)
    
#     return d_mu, d_sigma, d_theta

def gaussian_2d(X, Y, mu, sigma, theta):
    """2D Gaussian with rotation"""
    Sigma = get_covariance_matrix(sigma, theta)
    Sigma_inv = np.linalg.inv(Sigma)
    
    # Reshape X and Y into coordinate pairs
    coords = np.dstack((X - mu[0], Y - mu[1]))
    
    # Calculate quadratic form for each point
    quad = np.sum(coords @ Sigma_inv * coords, axis=2)
    
    return (1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))) * np.exp(-0.5 * quad)

if __name__ == "__main__":
    # Target parameters
    target_mu = np.array([3.0, 2.0])
    target_sigma = np.array([2.0, 1.5])
    target_theta = np.pi/4  # 45 degrees rotation

    # Initialize parameters
    mu = np.array([0.0, 0.0])
    sigma = np.array([1.0, 1.0])
    theta = 0.0

    learning_rate = 0.1
    num_iterations = 200

    loss_history = []
    mu_history = []
    sigma_history = []
    theta_history = []

    for i in range(num_iterations):
        loss = kl_divergence_2d(mu, sigma, theta, target_mu, target_sigma, target_theta)
        loss_history.append(loss)
        mu_history.append(mu.copy())
        sigma_history.append(sigma.copy())
        theta_history.append(theta)
        
        d_mu, d_sigma, d_theta = gradients_2d(mu, sigma, theta, target_mu, target_sigma, target_theta)
        
        # Update parameters
        mu = mu - learning_rate * d_mu
        sigma = sigma - learning_rate * d_sigma
        theta = theta - learning_rate * d_theta
        
        # Ensure sigma remains positive
        sigma = np.maximum(sigma, 1e-6)
        
        print(f"Iteration {i+1:03d}: Loss = {loss:.6f}")
        print(f"mu = [{mu[0]:.6f}, {mu[1]:.6f}]")
        print(f"sigma = [{sigma[0]:.6f}, {sigma[1]:.6f}]")
        print(f"theta = {theta:.6f}")

    # Update plotting code to include rotation
    # ... rest of the plotting code remains similar, but use the new gaussian_2d function ...

    # Plotting
    # 1. Loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="KL Divergence Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Gradient Descent Optimization of 2D Gaussian Parameters")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Contour plots at different iterations
    x = np.linspace(-5, 10, 100)
    y = np.linspace(-5, 10, 100)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(15, 10))
    plot_iterations = [0, 50, 99, 199]  # Initial, intermediate, and final states

    for idx, iter_num in enumerate(plot_iterations, 1):
        plt.subplot(2, 2, idx)
        
        # Plot target distribution
        Z_target = gaussian_2d(X, Y, target_mu, target_sigma, target_theta)
        plt.contour(X, Y, Z_target, colors='k', linestyles='--', levels=5, alpha=0.5)
        
        # Plot current distribution
        Z_current = gaussian_2d(X, Y, mu_history[iter_num], sigma_history[iter_num], theta_history[iter_num])
        plt.contour(X, Y, Z_current, colors='r', levels=5)
        
        plt.title(f'Iteration {iter_num+1}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 3. 3D surface plot of final state
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    Z_target = gaussian_2d(X, Y, target_mu, target_sigma, target_theta)
    Z_final = gaussian_2d(X, Y, mu_history[-1], sigma_history[-1], theta_history[-1])

    ax.plot_surface(X, Y, Z_target, cmap='viridis', alpha=0.5)
    ax.plot_surface(X, Y, Z_final, cmap='plasma', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Probability Density')
    ax.set_title('Final 2D Gaussian Distribution\n(Target vs Optimized)')

    plt.show()