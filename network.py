import numpy as np
import torch.nn as nn
import torch

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Yolo(nn.Module):
    '''
    
    '''

    def __init__(self):
        super(Yolo, self).__init__()
        #self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.leaky1 = nn.LeakyReLU(0.1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.leaky2 = nn.LeakyReLU(0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block1 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),

            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),

            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),

            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )

        self.final_h = 6
        self.final_w = 8
        downsample_size = self.final_h * self.final_w * 1024
        out_size = self.final_h * self.final_w * 14

        self.flat = Flatten()

        self.fc1 = nn.Linear(downsample_size, 4096)
        self.leaky3 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(4096, out_size)
        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()

        self.conv1.weight = torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2.weight = torch.nn.init.kaiming_normal_(self.conv2.weight)
        for layer in self.block1:
            if isinstance(layer, nn.Conv2d):
                layer.weight = torch.nn.init.kaiming_normal_(layer.weight)

        for layer in self.block2:
            if isinstance(layer, nn.Conv2d):
                layer.weight = torch.nn.init.kaiming_normal_(layer.weight)

        for layer in self.block3:
            if isinstance(layer, nn.Conv2d):
                layer.weight = torch.nn.init.kaiming_normal_(layer.weight)

        for layer in self.block4:
            if isinstance(layer, nn.Conv2d):
                layer.weight = torch.nn.init.kaiming_normal_(layer.weight)


    def forward (self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky1(out)
        out = self.maxpool1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leaky2(out)
        out = self.maxpool2(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)  # out -> [N, 1024, 6, 8] (N, C, H, W)

        out = self.flat(out)

        out = self.fc1(out)
        out = self.leaky3(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out.view(x.shape[0], self.final_h, self.final_w, 14)

# test = torch.zeros((10, 3, 341, 512), dtype=torch.float32)
# test_net = Yolo()
# out = test_net(test)
# print(out.shape)
