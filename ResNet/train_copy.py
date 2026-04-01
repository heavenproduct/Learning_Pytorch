import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import torchvision.models.resnet

from model import resnet34

def main():
    computer = os.uname()[0]
    if computer == 'Darwin':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    image_path = os.path.join(data_root, 'data_set', 'flower_data')
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image path {image_path} does not exist')

    # 定义训练集和测试集
    batch_size = 16

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                         transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                       transform=data_transform['val'])

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 1, 8])
    print(f'Using {nw} dataloader workers every process')

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=nw)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=nw)

    print(f'Using {train_num} images for training, {val_num} images for validation')

    # 获取文件夹中的分类(先存入字典,然后将字典中数据存入 json)
    flower_list = train_dataset.class_to_idx
    cla_list = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_list, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 这里不能单纯的修改 num_classes=5, 因为要加载预训练参数,
    # 加载后需要修改全连接层,这样模型参数的尺寸才能匹配
    net = resnet34()
    # 这里使用预训练好的参数,加快训练(网络上下载)
    model_weight_path = './resnet34-pre.pth'
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f'Model weight path {model_weight_path} does not exist')

    missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)

    # 这里是必要的,修改全连接层,使得从 1000 -> 5, 符合我们的需要
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    best_acc = 0.0
    save_path = './resNet34.pth'
    for epoch in range(3):
        # 训练
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = f'train epoch [{epoch+1}/3], loss: {loss:.3f}'

        # 验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                output = net(val_images)
                predict_y = torch.max(output, dim=1)[1]
                acc += (predict_y == val_labels).sum().item()
            val_accurate = acc / val_num

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print(f'{epoch+1} train_loss: {running_loss/step:.3f} test_accuracy:{val_accurate*100:.2f}%')

    print('Finished Training')

if __name__ == '__main__':
    main()
