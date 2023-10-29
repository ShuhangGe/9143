import torch.nn as nn
import torch.nn.functional as F

class LeNet5_BatchNorm_2(nn.Module):
    def __init__(self):
        super(LeNet5_BatchNorm_2, self).__init__()

        # Convolutional layers
        
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
        self.bn1 = nn.BatchNorm2d(6)     # BatchNorm after 1st conv layer

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)    # BatchNorm after 2nd conv layer
        self.bn3 = nn.BatchNorm1d(120)
        self.bn4 = nn.BatchNorm1d(84)
        # Fully connected layers
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes for MNIST

    def forward(self, x):
        
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)  # flatten

        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)

        return x

class LeNet5_BatchNorm_3(nn.Module):
    def __init__(self):
        super(LeNet5_BatchNorm_3, self).__init__()

        # Convolutional layers
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
        self.bn1 = nn.BatchNorm2d(6)     # BatchNorm after 1st conv layer

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)    # BatchNorm after 2nd conv layer
        self.bn3 = nn.BatchNorm1d(120)
        self.bn4 = nn.BatchNorm1d(84)
        # Fully connected layers
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = self.bn0(x)
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)  # flatten

        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)

        return x



class LeNet5_Dropout(nn.Module):
    def __init__(self):
        super(LeNet5_Dropout, self).__init__()

        self.dropout_input = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.dropout_input(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        x = self.dropout3(F.relu(self.fc1(x)))
        x = self.dropout4(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

class LeNet5_Dropout_bn(nn.Module):
    def __init__(self):
        super(LeNet5_Dropout_bn, self).__init__()

        self.dropout_input = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(6) 
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm1d(120)

        self.bn4 = nn.BatchNorm1d(84)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.dropout_input(x)
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        x = self.dropout3(F.relu(self.bn3(self.fc1(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc2(x))))
        x = self.fc3(x)

        return x
