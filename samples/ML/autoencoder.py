import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
HIDDEN_DIM = 128
LATENT_DIM = 2
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 20

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 28 * 28),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def train_autoencoder(data_dir='./data/datasets/mnist', device='cuda'):
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model and optimizer
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data.view(-1, 28*28))
            
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}')
    
    return model

def visualize_latent_space(model, data_dir='./data/datasets/mnist', device='cuda'):
    # Load test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            _, latent = model(data)
            latent_vectors.append(latent.cpu())
            labels.append(label)
    
    latent_vectors = torch.cat(latent_vectors, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Create scatter plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('MNIST Digits in 2D Latent Space')
    plt.xlabel('First Latent Dimension')
    plt.ylabel('Second Latent Dimension')
    plt.savefig('./data/mid/SailAlgo/mnist_latent_space.png')
    plt.show()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train model
    model = train_autoencoder(device=device)
    
    # Visualize latent space
    visualize_latent_space(model, device=device)
    
    # Save model
    torch.save(model.state_dict(), './data/mid/SailAlgo/autoencoder_mnist.pth')

if __name__ == "__main__":
    main()