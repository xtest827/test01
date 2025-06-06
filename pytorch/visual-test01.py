import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch.utils.data
import pytorch_lightning as pl
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy  # For binary accuracy

# Define hyperparameters
image_size = 64
batch_size = 256

# Define data paths
data_path_train = "cat-and-dog/training_set/training_set"
data_path_test = "cat-and-dog/test_set/test_set"

# Define transformations
transform = T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(root=data_path_train, transform=transform)
test_dataset = ImageFolder(root=data_path_test, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

class ImageClassifier(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()  # Fixed typo: Relu -> ReLU
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv_layer2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # After two 2x2 pooling layers on 64x64 input: 64 -> 32 -> 16
        self.fully_connected_1 = nn.Linear(in_features=6 * 16 * 16, out_features=1000)
        self.fully_connected_2 = nn.Linear(in_features=1000, out_features=250)
        self.fully_connected_3 = nn.Linear(in_features=250, out_features=60)
        self.fully_connected_4 = nn.Linear(in_features=60, out_features=2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        x = self.conv_layer(input)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv_layer2(x)
        x = self.relu2(x)
        x = self.pool(x)  # Added second pooling to match dimensions
        x = x.view(-1, 6 * 16 * 16)  # Fixed dimensions: 6 channels * 16x16
        x = self.fully_connected_1(x)
        x = F.relu(x)
        x = self.fully_connected_2(x)
        x = F.relu(x)
        x = self.fully_connected_3(x)
        x = F.relu(x)
        x = self.fully_connected_4(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)  # Fixed typo: Adan -> Adam
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        acc = accuracy(outputs, targets, task="binary")  # Fixed method name and added task
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        acc = accuracy(outputs, targets, task="binary")
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        return {"test_loss": loss, "test_accuracy": acc}

# Initialize model and trainer
model = ImageClassifier()
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # Replaced gpus=1
    devices=1,
    progress_bar_refresh_rate=30  # Deprecated, use callbacks instead if needed
)

# Train and test
trainer.fit(model, train_dataloaders=train_loader)  # Fixed "train" string to train_loader
trainer.test(model, dataloaders=test_loader)  # Fixed test_dataloaders name and added model