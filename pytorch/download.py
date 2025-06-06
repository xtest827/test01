import torch
from torchvision import models

model = models.inception_v3(weights='IMAGENET1K_V1')
