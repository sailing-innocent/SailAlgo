import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from args import ParamGroup

# Hyperparameters
LATENT_DIM = 100
HIDDEN_DIM = 256
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
EPOCHS = 50
BETA1 = 0.5

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 28 * 28),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(28 * 28, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        x = img.view(-1, 28 * 28)
        return self.model(x)

def train_gan(data_dir='./data/datasets/mnist', device='cuda'):
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(EPOCHS):
        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            label_real = torch.ones(batch_size, 1).to(device)
            label_fake = torch.zeros(batch_size, 1).to(device)
            
            output_real = discriminator(real_imgs)
            d_loss_real = criterion(output_real, label_real)
            
            z = torch.randn(batch_size, LATENT_DIM).to(device)
            fake_imgs = generator(z)
            output_fake = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(output_fake, label_fake)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_imgs)
            g_loss = criterion(output_fake, label_real)
            g_loss.backward()
            g_optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')
    
    return generator, discriminator

def generate_samples(generator, num_samples=10, device='cuda'):
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, LATENT_DIM).to(device)
        samples = generator(z)
        samples = (samples + 1) / 2  # Denormalize
        samples = samples.cpu()
        
        # Plot generated digits
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*2, 2))
        for i, ax in enumerate(axes):
            ax.imshow(samples[i].squeeze(), cmap='gray')
            ax.axis('off')
            plt.imsave(f'./data/mid/SailAlgo/gan_mnist_{i}.png', 
                      samples[i].squeeze(), cmap='gray')
        plt.show()

class Params(ParamGroup):
    def __init__(self, parser):
        self.skip_training = False
        self.name = "gan_mnist"
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
        generator, _ = train_gan(data_dir, device)
        # Save the generator
        torch.save(generator.state_dict(), save_path)
    else:
        # Load pre-trained generator
        generator = Generator()
        generator.to(device)
        generator.load_state_dict(torch.load(save_path))

    # Generate sample digits
    generate_samples(generator, device=device)