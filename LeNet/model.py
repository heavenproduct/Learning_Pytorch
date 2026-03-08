# 参考笔记: https://blog.csdn.net/m0_37867091/article/details/107136477
import torch
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.Linear1 = nn.Linear(32 * 5 * 5, 120)
        self.Linear2 = nn.Linear(120, 84)
        self.Linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 5 * 5) # 此步骤之前的数据依旧是一个 3 维的 Tensor,所以需要展平才能传到全连接层
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x

if __name__ == '__main__':
    net = LeNet()
    the_input = torch.ones((64, 3, 32, 32))
    the_output = net(the_input)
    print(the_output.shape)