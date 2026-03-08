import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载图像
im = Image.open('catt.jpeg')
im = transform(im)  # 形状: [3, 32, 32]
im = torch.unsqueeze(im, dim=0)  # 形状: [1, 3, 32, 32] (增加batch维度)

# 实例化网络并加载参数
net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))
net.eval()  # 设置为评估模式，这会影响 Dropout、BatchNorm 等层的行为

# 类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 预测
with torch.no_grad():
    outputs = net(im)
    # 获取最大概率对应的索引（保持为 tensor）
    predict = torch.max(outputs, dim=1)[1]
    # 或者可以这么写
    # _, predict = predict.max(dim=1)

# 打印结果（使用 .item() 安全转换）
print(f"预测类别: {classes[predict.item()]}")