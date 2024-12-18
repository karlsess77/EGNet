# 这部分代码是为了给模型信息提供类别约束
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt
import seaborn as sns

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Classifier_reg(nn.Module):
    def __init__(self, in_ch,  num_classes, H):
        super(Classifier_reg, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch * 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_ch * 2, out_channels=in_ch * 4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_ch * 4 * (H // 4) * (H // 4), 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x= GRL.apply(x, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        softmax_output = F.softmax(x, dim=1)


        max_indices = torch.argmax(softmax_output, dim=1)

        one_hot = torch.zeros_like(softmax_output)

        one_hot.scatter_(1, max_indices.unsqueeze(1), 1)
        return one_hot


