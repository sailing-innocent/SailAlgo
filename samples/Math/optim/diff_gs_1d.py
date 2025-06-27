# -*- coding: utf-8 -*-
# @file diff_gs_1d.py
# @brief Differential Gaussian 1D
# @author sailing-innocent
# @date 2025-02-23
# @version 1.0
# ---------------------------------

import numpy as np
import matplotlib.pyplot as plt

# KL divergence from target Gaussian to candidate Gaussian:
# D_KL = log(sigma/target_sigma) + (target_sigma^2 + (target_mu-mu)^2) / (2*sigma^2) - 0.5
def kl_divergence(mu, sigma, target_mu, target_sigma):
    return np.log(sigma / target_sigma) + (target_sigma**2 + (target_mu - mu)**2) / (2 * sigma**2) - 0.5

# Compute gradients of KL divergence with respect to mu and sigma
def gradients(mu, sigma, target_mu, target_sigma):
    d_mu = - (target_mu - mu) / (sigma**2)
    d_sigma = 1 / sigma - (target_sigma**2 + (target_mu - mu)**2) / (sigma**3)
    return d_mu, d_sigma

if __name__ == "__main__":
    # Target Gaussian parameters
    target_mu = 3.0
    target_sigma = 2.0

    # Initialize candidate parameters
    mu = 0.0
    sigma = 1.0

    learning_rate = 0.2
    num_iterations = 100

    loss_history = []
    mu_history = []
    sigma_history = []

    for i in range(num_iterations):
        loss = kl_divergence(mu, sigma, target_mu, target_sigma)
        loss_history.append(loss)
        mu_history.append(mu)
        sigma_history.append(sigma)
        
        d_mu, d_sigma = gradients(mu, sigma, target_mu, target_sigma)
        
        # Update parameters
        mu = mu - learning_rate * d_mu
        sigma = sigma - learning_rate * d_sigma
        
        # Ensure sigma remains positive
        sigma = max(sigma, 1e-6)
        
        print(f"Iteration {i+1:03d}: Loss = {loss:.6f}, mu = {mu:.6f}, sigma = {sigma:.6f}")

    # Plotting KL divergence loss over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="KL Divergence Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Gradient Descent Optimization of 1D Gaussian Parameters")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Plot candidate Gaussian curves every 30 iterations alongside the target Gaussian
    plt.figure(figsize=(10, 6))
    x = np.linspace(-5, 10, 300)
    # Compute target Gaussian PDF: 1/(σ√(2π)) exp(-0.5*((x-μ)/σ)²)
    target_pdf = 1/(target_sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - target_mu)/target_sigma)**2)
    plt.plot(x, target_pdf, 'k--', label='Target Gaussian')
    # plot initial state
    initial_pdf = 1/(sigma_history[0] * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu_history[0])/sigma_history[0])**2)
    plt.plot(x, initial_pdf, label='Initial State')

    # Plot candidate curves for selected iterations (every 30 iterations)
    for idx in range(29, len(mu_history), 30):
        mu_candidate = mu_history[idx]
        sigma_candidate = sigma_history[idx]
        candidate_pdf = 1/(sigma_candidate * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu_candidate)/sigma_candidate)**2)
        plt.plot(x, candidate_pdf, label=f'Iteration {idx+1}')

    # Optionally, also plot the final iteration if not already included
    if (len(mu_history) - 1) % 30 != 0:
        idx = len(mu_history) - 1
        mu_candidate = mu_history[idx]
        sigma_candidate = sigma_history[idx]
        candidate_pdf = 1/(sigma_candidate * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu_candidate)/sigma_candidate)**2)
        plt.plot(x, candidate_pdf, label=f'Iteration {idx+1}')

    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("Evolution of Candidate Gaussian Curve Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()