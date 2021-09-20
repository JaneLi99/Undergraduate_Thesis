import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文

# 图片太大超过限制，PIL处理不了，必须把图片删除一部分，导致报error
# 如果设为true，此时加载的图片会少掉一部分，但是在大数据加载一张半张残缺图像是应该没大影响
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transformer = transforms.Compose([
  transforms.Resize((150, 150)),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize([0.5,0.5,0.5],  #0-1 to [-1, 1], formula (x - mean) / std
            [0.5,0.5,0.5])
])

#Dataloader
train_path = 'D:/李佳明/北印/grad_des_proj/grad_des_proj/image_train'
test_path = 'D:/李佳明/北印/grad_des_proj/grad_des_proj/image_test'

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform = transformer),
    batch_size = 4, shuffle = True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform = transformer),
    batch_size = 4, shuffle = True
)

#categories
root = pathlib.Path(train_path)
classes = sorted(j.name.split('/')[-1] for j in root.iterdir())
print(classes)

#CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        # output size after convolution filter
        #((w - f + 2P) / s) + 1

        # input shape = (256, 3, 150, 150)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # 输入图片的高度，输出图片高度，3*3的卷积核，卷积核在图上滑动，每隔一个扫一次，图外边补零
        #Shape = (256, 12, 150, 150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape = (256, 12, 150, 150)
        # 经过卷积层 输出[12, 150, 150] 传入池化层
        self.relu1 = nn.ReLU()
        # Shape = (256, 12, 150, 150)

        self.pool = nn.MaxPool2d(kernel_size = 2)
        # 经过池化 输出[12, 75, 75] 传入下一个卷积

        #Reduce the image size be factor 2
        #Shape = (256, 12, 75, 75)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape = (256, 20, 75, 75)
        self.relu2 = nn.ReLU()
        # Shape = (256, 20, 75, 75)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape = (256, 32, 75, 75)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape = (256, 32, 75, 75)
        self.relu3 = nn.ReLU()
        # Shape = (256, 32, 75, 75)

        self.fc = nn.Linear(in_features = 32*75*75, out_features = num_classes)

    #Feed forward function
    def forward(self, *input):
        output = self.conv1(input[0])
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        #Above output will be i matrix from, with shape(256, 32, 75, 75)
        output = output.view(-1, 32 * 75 * 75)

        output = self.fc(output)

        return output


model = ConvNet(num_classes=2).to(device)

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 50

train_count = len(glob.glob(train_path + '/**/*.jpg'))
test_count = len(glob.glob(test_path + '/**/*.jpg'))
print(train_count, test_count)

best_accuracy = 0.0

epoch_list = []
train_accuracy_list = []
train_loss_list = []
test_accuracy_list = []
test_loss_list = []
for epoch in range(num_epochs):
    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(images)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(output.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation and testing dataset
    model.eval()
    test_accuracy = 0.0
    test_loss = 0.0
    for i, (images, labels) in enumerate(test_loader):
        output = model(images)
        _, prediction = torch.max(output.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))
        # loss = loss_function(output, labels)
        # loss.backward()
        # optimizer.step()
        #
        # test_loss += loss.cpu().data * images.size(0)
        # _, prediction = torch.max(output.data, 1)

    test_accuracy = test_accuracy / test_count
    # test_loss = train_loss / train_count

    epoch_list.append(int(epoch))
    train_accuracy_list.append(float(train_accuracy))
    train_loss_list.append(float(train_loss))
    test_accuracy_list.append(float(test_accuracy))
    # test_loss_list.append(float(test_loss))

    print('Epoch: ' + str(epoch) + ', Train Loss: ' + str(float(train_loss)) +', Train Accuracy: ' + str(train_accuracy)
          + ', Test Accuracy: ' + str(test_accuracy))

    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint4.model')
        best_accuracy = test_accuracy

print('epoch_list', epoch_list)
print('train_accuracy_list', train_accuracy_list)
print('train_loss_list', train_loss_list)
print('test_accuracy_list', test_accuracy_list)
