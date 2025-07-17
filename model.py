import torch.nn as nn
import torch.nn.functional as F

class KeypointModel(nn.Module):
    def __init__(self):
        super(KeypointModel, self).__init__()
        
        # Сверточные слои
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Слой Max-pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_drop = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(512, 8)

    def forward(self, x):
        # Добавляем канальное измерение к входному изображению
        x = x.view(x.size(0), 1, 96, 96)
        
        # Сверточные слои с LeakyReLU, pooling и batch norm
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        
        # Выравниваем изображение для полносвязных слоев
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои
        x = F.leaky_relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        return x
