# model.py
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(CNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1      
        )
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional block
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1      
        )
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional block
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1      
        )
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2       
        )

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=dropout_rate)
     
        # Fully connected layer
        # After 3 conv layers with pooling: 32x32 -> 16x16 -> 8x8 -> 4x4
        # 128 channels * 4 * 4 = 2048 features
        self.fc = nn.Linear(
            in_features=128 * 4 * 4,
            out_features=num_classes   # 10 logits for 10 classes
        )

    def forward(self, x):
      
        x = self.conv1(x)      # (B, 3, 32, 32) -> (B, 32, 32, 32)
        x = self.bn1(x)         # Batch normalization
        x = F.relu(x)
        x = self.pool(x)       # (B, 32, 16, 16)

        x = self.conv2(x)      # (B, 32, 16, 16) -> (B, 64, 16, 16)
        x = self.bn2(x)         # Batch normalization
        x = F.relu(x)
        x = self.pool(x)       # (B, 64, 8, 8)

        x = self.conv3(x)      # (B, 64, 8, 8) -> (B, 128, 8, 8)
        x = self.bn3(x)         # Batch normalization
        x = F.relu(x)
        x = self.pool(x)       # (B, 128, 4, 4)

        # Flatten
        x = x.view(x.size(0), -1)   # (B, 128*4*4) = (B, 2048)
        # Apply dropout before fully connected layer (only during training)
        x = self.dropout(x)

        logits = self.fc(x)         # (B, num_classes)
        return logits
