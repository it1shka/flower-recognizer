import torch
from torch import nn


class ConvUnit(nn.Module):
  '''
  Convolutional Unit consisting of
  - Convolutional Layer
  - ReLU Activation
  - Batch Normalization
  - Max Pooling
  '''
  def __init__(self, in_channels: int, out_channels: int, conv_kernel: int = 3, pool_kernel: int = 2) -> None:
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, padding=1)
    self.relu = nn.ReLU()
    self.batch_norm = nn.BatchNorm2d(out_channels)
    self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    x = self.relu(x)
    x = self.batch_norm(x)
    x = self.pool(x)
    return x
  

class DenseUnit(nn.Module):
  '''
  Dense Unit consisting of
  - Linear Layer
  - ReLU Activation
  - Batch Normalization
  - Dropout
  '''
  def __init__(self, in_features: int, out_features: int, droupout: float = 0.0, normalization: bool = True) -> None:
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)
    self.relu = nn.ReLU()
    self.normalization = normalization
    if normalization:
      self.batch_norm = nn.BatchNorm1d(out_features)
    self.dropout = nn.Dropout(droupout)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.linear(x)
    x = self.relu(x)
    if self.normalization:
      x = self.batch_norm(x)
    x = self.dropout(x)
    return x


class FlowerCNN(nn.Module):
  '''
  Convolutional Neural Network for classifying 5 types of flowers:
  - Lilly
  - Lotus
  - Orchid
  - Sunflower
  - Tulip

  Takes RGB images 224x224 pixels as input turned into tensors
  '''
  def __init__(self) -> None:
    super().__init__()
    self.extractor = nn.Sequential(
      ConvUnit(3, 32),
      ConvUnit(32, 64),
      ConvUnit(64, 128),
      ConvUnit(128, 256),
    )
    self.flatten = nn.Flatten()
    self.classifier = nn.Sequential(
      DenseUnit(50176, 1024, droupout=0.2),
      DenseUnit(1024, 256, droupout=0.2),
      DenseUnit(256, 64, droupout=0.2),
      DenseUnit(64, 5),
    )
    self.output_function = nn.Softmax(dim=1)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.extractor(x)
    x = self.flatten(x)
    x = self.classifier(x)
    x = self.output_function(x)
    return x
