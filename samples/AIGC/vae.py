import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from args import ParamGroup
import os 


# Hyperparameters
LATENT_DIM = 20
HIDDEN_DIM = 400
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 50

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_var = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 28 * 28),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train_vae(data_dir='./data/datasets/mnist', device='cuda'):
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        avg_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}')
    
    return model

def generate_digits(model, num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        # Sample from normal distribution
        z = torch.randn(num_samples, LATENT_DIM).to(device)
    
        samples = model.decode(z)
        samples = samples.cpu().view(-1, 28, 28)
        
        # Plot generated digits
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*2, 2))
        for i, ax in enumerate(axes):
            ax.imshow(samples[i], cmap='gray')
            ax.axis('off')
            plt.imsave(f'./data/mid/SailAlgo/vae_mnist_{i}.png', samples[i], cmap='gray')
        plt.show()


class Params(ParamGroup):
    def __init__(self, parser):
        self.skip_training = False
        self.name = "vae_mnist" # Name of the Experiment
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
    save_path = os.path.join(save_dir,  f"{name}.pth")

    if not skip_training:
    # Train the model
        model = train_vae(data_dir, device)
    else:
        model = VAE()
        model.to(device)
        model.load_state_dict(torch.load(save_path))
    # Generate sample digits
    generate_digits(model)
    # Save the model
    if not skip_training:
        torch.save(model.state_dict(), save_path)