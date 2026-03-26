import os
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from Learning_for_tudui.CNN_basic_number import batch_size
from model import GoogLeNet

def main():
    # 使用 GPU 训练, 自动判断是 Mac的显卡 或者 NVIDIA 显卡
    computer = os.uname()
    # print(computer[0])
    if computer[0] == 'Darwin':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 定义数据处理方式
    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # 定义数据(图片地址)
    data_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    image_path = os.path.join(data_root, 'data_set', 'flower_data')
    # 如果该地址不存在指定的文件夹, 升起警报并退出程序
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'{image_path} does not exist')

    # 定义数据集(数据集的地址和数据处理方式)和载入数据集, 得到数据集内部数据量
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                         transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                       transform=data_transform['val'])

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=nw)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=nw)


    # 这一步就直接自动将文件夹的名字创建成字典了
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    # 交换 key 和 value 的顺序,方便后续使用
    cla_dict = dict((val, key) for key, val in flower_list.items())

    # 将准备好的数据存入 json, dumps 是格式转换
    json_str = json.dumps(flower_list, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    print(f'using {train_num} images for training, using {val_num} images for validation')

    net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    epochs = 30
    best_acc = 0.0
    save_path = './googleNet.pth'
    train_steps = len(train_loader)

    for epoch in range(epochs):

        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            # 这里 因为 GoogleLenet 使用了两个辅助分类器(总共 3 个分类器)
            # 所以数据要特殊处理
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            loss0 = loss_function(logits,labels.to(device))
            loss1 = loss_function(aux_logits1,labels.to(device))
            loss2 = loss_function(aux_logits2,labels.to(device))
            loss = loss0 + loss1 + loss2
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = f'train epoch [{epoch+1}/{epochs}], loss{loss:.3f}'

        # validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

    if __name__ == '__main__':
        main()