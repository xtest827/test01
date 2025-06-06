import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 参数设置
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 10
batch_size = 128
learning_rate = 0.001

# 创建保存图像的文件夹
if not os.path.exists('../vae_samples'):
    os.makedirs('../vae_samples')

# 加载 MNIST 数据
dataset = torchvision.datasets.MNIST(root='data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)       # mu
        self.fc3 = nn.Linear(h_dim, z_dim)       # log_var
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    for i, (x, _) in enumerate(data_loader):
        x = x.view(-1, image_size).to(device)
        x_reconst, mu, log_var = model(x)

        # 重构损失 + KL散度损失
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = reconst_loss + kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每轮结束保存重建图像
    with torch.no_grad():
        x = x.view(-1, image_size).to(device)
        x_reconst, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28)[:8], x_reconst.view(-1, 1, 28, 28)[:8]], dim=0)
        save_image(x_concat, f'./vae_samples/recon_epoch{epoch + 1}.png', nrow=8)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.2f}')

# 显示最后一张图片
final_img_path = f'./vae_samples/recon_epoch{num_epochs}.png'
if os.path.exists(final_img_path):
    img = mpimg.imread(final_img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Reconstructed Images (原图在上, 重建图在下)")
    plt.show()
else:
    print("还没有图片生成，请检查训练过程是否执行。")
