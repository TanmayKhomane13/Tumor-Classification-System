import torch
import torch.nn as nn

# ====================== CNN ===========================
class CNN(nn.Module):
    def __init__(self, out_1 = 16, out_2 = 32, out_3 = 64):
        super(CNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = out_1, kernel_size = 3, padding = 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)

        self.cnn2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 3, padding = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.cnn3 = nn.Conv2d(in_channels = out_2, out_channels = out_3, kernel_size = 3, padding = 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)

        self.fc1 = nn.Linear(out_3 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 1)
    
    # prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)

        x = self.cnn3(x)
        x = torch.relu(x)
        x = self.maxpool3(x)
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
