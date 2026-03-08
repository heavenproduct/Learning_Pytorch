# 参考笔记: https://blog.csdn.net/m0_37867091/article/details/107136477

import torch
import torchvision
import torch.nn as nn
from model import *
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 导入 CIFAR10 数据集, 训练集为 50000 张图片,测试集为 10000 张图片
batch_size = 50
train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=10000, shuffle=False)

# 获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)
test_image, test_label = next(test_data_iter)

# 移动到设备
test_image = test_image.to(device)
test_label = test_label.to(device)

net = LeNet().to(device)  # 定义训练的网络模型
loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器（训练参数，学习率）

for epoch in range(5):  # 一个epoch即对整个训练集进行一次训练
    running_loss = 0.0
    time_start = time.perf_counter()

    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
        inputs, labels = data  # 获取训练集的图像和标签
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # 清除历史梯度

        # forward + backward + optimize
        outputs = net(inputs)  # 正向传播
        loss = loss_function(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数

        # 打印耗时、损失、准确率等数据
        running_loss += loss.item()

        if step % 1000 == 999:  # print every 1000 mini-batches，每1000步打印一次
            with torch.no_grad():  # 在以下步骤中（验证过程中）不用计算每个节点的损失梯度，防止内存占用
                outputs = net(test_image)  # 测试集传入网络（test_batch_size=10000），output维度为[10000,10]
                predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出,这就是[1]的作用,[0]对应的是最大值
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                      (epoch + 1, step + 1, running_loss / 500, accuracy))

                print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
                running_loss = 0.0

print('Finished Training')

# 保存训练得到的参数
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)


