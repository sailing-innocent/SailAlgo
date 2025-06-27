# -*- coding: utf-8 -*-
# @file vqvae.py
# @brief VQ-VAE
# @author sailing-innocent
# @date 2025-02-15
# @version 1.0
# ---------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from args import ParamGroup

# Hyperparameters
NUM_EMBEDDINGS = 512
EMBEDDING_DIM = 64
HIDDEN_DIM = 256
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 50
COMMITMENT_COST = 0.25

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + COMMITMENT_COST * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity

class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, HIDDEN_DIM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_DIM, EMBEDDING_DIM, kernel_size=3, stride=1, padding=1)
        )
        
        self.vector_quantization = VectorQuantizer(NUM_EMBEDDINGS, EMBEDDING_DIM)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(EMBEDDING_DIM, HIDDEN_DIM, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(HIDDEN_DIM, HIDDEN_DIM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(HIDDEN_DIM, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity = self.vector_quantization(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity

def train_vqvae(data_dir='./data/datasets/mnist', device='cuda'):
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, vq_loss, perplexity = model(data)
            recon_loss = F.mse_loss(recon_batch, data)
            loss = recon_loss + vq_loss
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        avg_loss = train_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f} Perplexity: {perplexity:.2f}')
    
    return model

def generate_samples(model, num_samples=10, device='cuda'):
    model.eval()
    
    with torch.no_grad():
        # Get the correct spatial dimensions from encoder output
        dummy_input = torch.zeros(1, 1, 28, 28).to(device)
        encoded = model.encoder(dummy_input)
        h, w = encoded.shape[2:]  # Get spatial dimensions from encoder output
        
        # Sample from prior (uniform over codebook)
        indices = torch.randint(0, NUM_EMBEDDINGS, (num_samples, h, w)).to(device)
        flat_indices = indices.view(-1)
        
        # Get embeddings and reshape
        quantized = model.vector_quantization.embedding(flat_indices)
        quantized = quantized.view(num_samples, h, w, EMBEDDING_DIM)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Generate samples
        samples = model.decoder(quantized)
        samples = samples.cpu()
        
        # Plot generated digits
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*2, 2))
        for i, ax in enumerate(axes):
            ax.imshow(samples[i].squeeze(), cmap='gray')
            ax.axis('off')
            plt.imsave(f'./data/mid/SailAlgo/vqvae_mnist_{i}.png', 
                      samples[i].squeeze(), cmap='gray')
        plt.show()

class Params(ParamGroup):
    def __init__(self, parser):
        self.skip_training = False
        self.name = "vqvae_mnist"
        self.data_dir = "data/datasets/mnist"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = "data/mid/SailAlgo/"
        super().__init__(parser, "Pipeline Parameters")

def experiment(params: Params):
    skip_training = params.skip_training
    data_dir = params.data_dir
    save_dir = params.save_dir
    device = params.device
    name = params.name
    save_path = os.path.join(save_dir, f"{name}.pth")

    if not skip_training:
        # Train the model
        model = train_vqvae(data_dir, device)
        # Save the model
        torch.save(model.state_dict(), save_path)
    else:
        # Load pre-trained model
        model = VQVAE()
        model.to(device)
        model.load_state_dict(torch.load(save_path))

    # Generate sample digits
    generate_samples(model, device=device)