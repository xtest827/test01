import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
latent_dim = 100
num_classes = 10
image_size = 28 * 28
epochs = 50
lr = 0.0002

# Transforms and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Scale images to [-1, 1]
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 784),  # ✅ Fixed size
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([z, label_input], 1)
        out = self.model(x)
        return out.view(x.size(0), 1, 28, 28)


# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 10)
        self.model = nn.Sequential(
            nn.Linear(784 + 10, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        label_input = self.label_emb(labels)
        x = torch.cat([x, label_input], 1)
        out = self.model(x)
        return out.view(-1)

# Initialize models
G = Generator().to(device)
D = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# TensorBoard writer
writer = SummaryWriter()

# Training loop
for epoch in range(epochs):
    for batch_idx, (real_images, labels) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        labels = labels.to(device)

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        fake_images = G(z, fake_labels)

        real_validity = D(real_images, labels)
        fake_validity = D(fake_images.detach(), fake_labels)

        real_loss = criterion(real_validity, torch.ones(batch_size).to(device))
        fake_loss = criterion(fake_validity, torch.zeros(batch_size).to(device))
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        fake_validity = D(fake_images, fake_labels)
        g_loss = criterion(fake_validity, torch.ones(batch_size).to(device))

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(train_loader)} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")
            writer.add_scalars('Loss', {'D': d_loss.item(), 'G': g_loss.item()}, epoch * len(train_loader) + batch_idx)

# Sample visualization after training
z = torch.randn(100, latent_dim).to(device)
labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).to(device)
images = G(z, labels)
grid = make_grid(images, nrow=10, normalize=True)

plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
plt.axis('off')
plt.title("Generated Digits")
plt.show()

# Function to generate a digit image
def generate_digit(generator, digit):
    z = torch.randn(1, latent_dim).to(device)
    label = torch.LongTensor([digit]).to(device)
    img = generator(z, label).detach().cpu()
    img = 0.5 * img + 0.5  # Rescale from [-1, 1] to [0, 1]
    return transforms.ToPILImage()(img.squeeze())

# Example: Show digit "8"
digit_img = generate_digit(G, 8)
digit_img.show()

# Don't forget to close TensorBoard writer
writer.close()
