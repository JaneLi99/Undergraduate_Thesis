"""
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
import pathlib
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文

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
train_path = 'C:/Users/lijiaming3/Documents/grad_des_proj/image_train'
test_path = 'C:/Users/lijiaming3/Documents/grad_des_proj/image_test'

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform = transformer),
    batch_size = 7, shuffle = True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform = transformer),
    batch_size = 7, shuffle = True
)

#categories
root = pathlib.Path(train_path)
classes = sorted(j.name.split('/')[-1] for j in root.iterdir())
print(classes)

#CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features = 32*75*75, out_features = num_classes)

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

    test_accuracy = test_accuracy / test_count

    print('Epoch: ' + str(epoch) + ', Train Loss: ' + str(float(train_loss)) +', Train Accuracy: ' + str(train_accuracy)
          + ', Test Accuracy: ' + str(test_accuracy))

    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint7.model')
        best_accuracy = test_accuracy
"""

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
x = np.arange(-10, 10)
y = np.where(x<0,0,x)#满足条件(condition)，输出x，不满足输出y
plt.xlim(-11,11)
plt.ylim(-11,11)
ax = plt.gca() # get current axis 获得坐标轴对象
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none') # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

plt.plot(x,y,label='ReLU',linestyle="-", color="darkviolet")#label为标签
plt.legend(['ReLU'])
plt.savefig('ReLU.jpg', dpi=500) #指定分辨
plt.show()