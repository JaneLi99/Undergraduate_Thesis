import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob

train_path = 'D:/李佳明/北印/grad_des_proj/grad_des_proj/image_train'
pred_path = 'D:/李佳明/北印/grad_des_proj/grad_des_proj/image_pred'

root = pathlib.Path(train_path)
classes = sorted(j.name.split('/')[-1] for j in root.iterdir())
print(classes)

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        # output size after convolution filter
        #((w - f + 2P) / s) + 1

        # input shape = (256, 3, 150, 150)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        #Shape = (256, 12, 150, 150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape = (256, 12, 150, 150)
        self.relu1 = nn.ReLU()
        # Shape = (256, 12, 150, 150)

        self.pool = nn.MaxPool2d(kernel_size = 2)

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

checkpoint = torch.load('best_checkpoint7.model')
model = ConvNet(num_classes = 2)
model.load_state_dict(checkpoint)
model.eval()

#Transforms
transformer = transforms.Compose([
  transforms.Resize((150, 150)),
  # transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize([0.5,0.5,0.5],  #0-1 to [-1, 1], formula (x - mean) / std
            [0.5,0.5,0.5])
])

#Prediction function
def prediction(img_path, transformer):
    image = Image.open(img_path).convert('RGB')
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    input = Variable(image_tensor)
    output = model(input)

    index = output.data.numpy().argmax()
    pred = classes[index]

    return pred

images_path = glob.glob(pred_path + '/*/*.jpg')

pred_dict = {}
pred_accuracy_list = []
#pred_accuracy = 0.0
for i in images_path:
    res = prediction(i, transformer)
    pred_dict[i[i.rfind('/') + 1:]] = res

    a = os.path.dirname(i)  # 先获取文件路径
    #print(a)
    image_name = os.path.basename(a)  # 从文件路径中读取最后一个文件夹的名字
    if str(res) == str(image_name):
        pred_accuracy_list.append(1)
    else:
        pred_accuracy_list.append(0)
    print('模型预测结果: ', res, '; 无人机实际型号: ', image_name)

pred_accuracy = np.sum(pred_accuracy_list) / len(pred_accuracy_list)
print('预测正确率: ', pred_accuracy)

#print(pred_dict)

