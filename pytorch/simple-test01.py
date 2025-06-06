import pytorch_lightning as pl
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import transforms

# Input and target tensors
inputs = [torch.tensor([0., 0.]),
          torch.tensor([0., 1.]),
          torch.tensor([1., 0.]),
          torch.tensor([1., 1.])]

targets = [torch.tensor([0.]),
           torch.tensor([1.]),
           torch.tensor([1.]),
           torch.tensor([0.])]

class XOR(pl.LightningModule):
    def __init__(self):
        super(XOR, self).__init__()
        self.input_layer = nn.Linear(2, 4)
        self.output_layer = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, input):
        x = self.input_layer(input)
        x = self.sigmoid(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        return loss

# Create dataset and dataloader
dataset = list(zip(inputs, targets))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model
model = XOR()

# Add ModelCheckpoint callback to save best model
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='train_loss',  # Monitor training loss
    dirpath='checkpoints/',  # Directory to save checkpoints
    filename='xor-{epoch:02d}-{train_loss:.2f}',
    save_top_k=1,  # Save only the best model
    mode='min',  # Minimize the monitored metric (loss)
)

# Initialize trainer with callback
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[checkpoint_callback],  # Add the checkpoint callback
    enable_checkpointing=True,  # Explicitly enable checkpointing
)

# Train the model (fixed the dataloaders parameter name)
trainer.fit(model, train_dataloaders=train_loader)  # Fixed from data_inputs_targets to train_loader

# Print the best model path
print(checkpoint_callback.best_model_path)

# Test the model
with torch.no_grad():
    for x, y in zip(inputs, targets):
        prediction = model(x)
        print(f"Input: {x.tolist()}, Target: {y.tolist()}, Prediction: {prediction.tolist()}")
