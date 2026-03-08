import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
import torch.optim as optim
from model import AlexNet
import os
import json

# 使用 GPU 训练, 自动判断是 Mac的显卡 或者 NVIDIA 显卡
computer = os.uname()
# print(computer[0])
if computer[0] == 'Darwin':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),        # 随机裁剪，再缩放成 224×224
        transforms.RandomHorizontalFlip(p=0.5),   # 水平方向随机翻转(镜像)，概率为 0.5, 即一半的概率翻转, 一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),            # cannot 224, must (224, 224)
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
}

# 获取图像数据集的路径
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))   # 拿到「当前脚本所在目录往上两级」的绝对路径
# print(data_root)
image_path = data_root + "/Learn_from_PiLiPaLa/AlexNet" + "/flower_data"

# 导入训练集并批量处理
# ImageFolder 会自动扫描文件夹, 读取 train/ 下的所有子文件夹名
# 按字母顺序（或系统顺序）给每个文件夹分配一个索引（从0开始,创建这个映射字典
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])
train_num =len(train_dataset)

train_loader = DataLoader(dataset = train_dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0)

# 导入验证集并进行预处理
validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
validata_num = len(validate_dataset)
validate_loader = DataLoader(dataset = validate_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=0)
# 字典,类别:索引{'daisy':0, 'dandelion':1, 'rose':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
# print(flower_list)
print(flower_list.items())
# 创建一个新的字典,将flower_list 中的 key 和 val 调换位置
cla_dict = dict((val, key) for key, val in flower_list.items())
# print(cla_dict)

# 将 cla_dict 写入 json文件中
# .dumps(),将 python 对象转化为 json 对象, indent=4 是在最左侧添加 4 个空格
json_str = json.dumps(cla_dict, indent=4)
# 打开或者创建一个叫 xxx.json 的文件, .write()写入
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# print(len(flower_list))
net = AlexNet(num_classes=len(flower_list), init_weights=True).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './AlexNet.pth'
best_acc = 0
