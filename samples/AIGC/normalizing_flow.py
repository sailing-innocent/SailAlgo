import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from args import ParamGroup

# Hyperparameters
INPUT_DIM = 784  # 28*28
HIDDEN_DIM = 256
NUM_FLOWS = 4
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 50

class PlanarFlow(nn.Module):
    def __init__(self, input_dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(input_dim))
        self.w = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.randn(1))
        
        # Ensure u'w > -1 (for invertibility)
        self.register_buffer('wu', None)
    
    def forward(self, z):
        # Compute u'w (recompute in case u or w have changed)
        wu = (self.w @ self.u).view(1)
        
        # Ensure invertibility using u_hat
        u_hat = self.u
        if wu < -1:
            u_hat = self.u + ((1 + wu) * self.w / torch.sum(self.w**2))
            
        # f(z) = z + u * h(w^T * z + b)
        lin = torch.matmul(z, self.w.view(-1, 1)) + self.b  # Shape: (batch_size, 1)
        activation = torch.tanh(lin)  # Shape: (batch_size, 1)
        z_out = z + u_hat.view(1, -1) * activation  # Shape: (batch_size, input_dim)
        
        # Calculate log determinant of Jacobian
        psi = (1 - activation**2) * self.w.view(1, -1)  # Shape: (batch_size, input_dim)
        log_det = torch.log(torch.abs(1 + torch.matmul(psi, u_hat.view(-1, 1))))  # Shape: (batch_size, 1)
        
        return z_out, log_det.squeeze()
class NormalizingFlow(nn.Module):
    def __init__(self, input_dim, num_flows):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList([PlanarFlow(input_dim) for _ in range(num_flows)])
        
    def forward(self, z):
        log_det_sum = 0
        
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det
            
        return z, log_det_sum

def train_flow(data_dir='./data/datasets/mnist', device='cuda'):
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = NormalizingFlow(INPUT_DIM, NUM_FLOWS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass through flows
            z, log_det = model(data)
            
            # Compute loss: negative log likelihood of the data
            prior_ll = -0.5 * torch.sum(z**2, dim=1) - 0.5 * INPUT_DIM * torch.log(torch.tensor(2 * torch.pi))
            loss = -(prior_ll + log_det).mean()
            
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}')
    
    return model

def generate_digits(model, num_samples=10, device='cuda'):
    model.eval()
    
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, INPUT_DIM).to(device)
        
        # Inverse flow to generate samples
        for flow in reversed(model.flows):
            # Approximate inverse using fixed-point iteration
            z_prev = z.clone()
            for _ in range(50):  # Number of iterations for inverse
                z_new = z_prev - (flow(z_prev)[0] - z)
                if torch.norm(z_new - z_prev) < 1e-4:
                    break
                z_prev = z_new
            z = z_prev
        
        samples = z.cpu().view(-1, 28, 28)
        
        # Plot generated digits
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*2, 2))
        for i, ax in enumerate(axes):
            ax.imshow(samples[i], cmap='gray')
            ax.axis('off')
            plt.imsave(f'./data/mid/SailAlgo/flow_mnist_{i}.png', 
                      samples[i], cmap='gray')
        plt.show()

class Params(ParamGroup):
    def __init__(self, parser):
        self.skip_training = False
        self.name = "flow_mnist"
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

    print(f"Training {name} model on {device} device with {data_dir} dataset and saving to {save_path}")

    if not skip_training:
        # Train the model
        model = train_flow(data_dir, device)
        # Save the model
        torch.save(model.state_dict(), save_path)
    else:
        # Load pre-trained model
        model = NormalizingFlow(INPUT_DIM, NUM_FLOWS)
        model.to(device)
        model.load_state_dict(torch.load(save_path))

    # Generate sample digits
    generate_digits(model, device=device)

if __name__ == "__main__":
    params = Params(None)
    experiment(params)