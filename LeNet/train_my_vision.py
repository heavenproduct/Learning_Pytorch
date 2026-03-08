# 参考笔记: https://blog.csdn.net/m0_37867091/article/details/107136477
# 按照我喜欢的方式修改了一下

from model import *
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

import time

# 首先检查设备,尽可能使用 gpu
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'We use device: {device}')

# 定义处理数据的方式
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载数据,训练集和测试集CIFAR10
batch_size = 50
train_dataset = datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = datasets.CIFAR10(root='../data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=0)

# 实例化模型
net = LeNet().to(device)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器（训练参数，学习率）

# 定义训练
def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 1000 == 999:
            print(f'In {epoch+1} training, {batch_idx+1} batches left, the loss is {running_loss/1000}')

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('测试完成~')
    print(f'Accuracy of the model on the test images: {accuracy} %')

if __name__ == '__main__':
    for epoch in range(10):
        strat_time = time.time()
        train(epoch)
        print('训练完成,开始测试')
        test()
        end_time = time.time()
        using_time = end_time - strat_time
        print(f'The training time is {using_time} seconds.')

        # 保存训练得到的参数
    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)



