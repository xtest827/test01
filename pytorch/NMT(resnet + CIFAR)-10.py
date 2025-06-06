import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# =========================
# 数据增强与预处理
# =========================
trans_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

trans_valid = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# 加载 CIFAR-10 数据集
# =========================
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=trans_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=trans_valid)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# =========================
# 加载预训练模型
# =========================
weights = ResNet18_Weights.DEFAULT
net = resnet18(weights=weights)

# 冻结所有参数
for param in net.parameters():
    param.requires_grad = False

# 替换最后全连接层（输出为10类）
net.fc = nn.Linear(512, 10)

# 使用GPU或CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# =========================
# 参数统计
# =========================
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"原总参数个数:{total_params}")
print(f"需要训练参数:{trainable_params}")

# =========================
# 损失函数 & 优化器
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.fc.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)

# =========================
# 训练函数定义
# =========================
def train(model, trainloader, testloader, epochs, optimizer, criterion):
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.3f}, 训练准确率: {train_acc:.2f}%")

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"→ 验证准确率: {val_acc:.2f}%\n")

# =========================
# 主函数入口（必须加）
# =========================
if __name__ == "__main__":
    train(net, trainloader, testloader, epochs=20, optimizer=optimizer, criterion=criterion)
