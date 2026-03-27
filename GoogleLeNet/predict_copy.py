import os
import json

import torch
from PIL import Image
from numpy.ma.core import squeeze
from torchvision import transforms
import matplotlib.pyplot as plt

from model import GoogLeNet

def main():
    # 使用 GPU 训练, 自动判断是 Mac的显卡 或者 NVIDIA 显卡
    computer = os.uname()
    # print(computer[0])
    if computer[0] == 'Darwin':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

    # load image
    img_path = './tulip.jpg'
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'file {img_path} does not exist')
    img = Image.open(img_path)
    plt.imshow(img)
    # 加载的图片是 [H, W, C] 的数据序列,需要转换结构才能使用
    # 转化为 [C, H, W]
    img = data_transform(img)
    # 网络模型需要有 batch, 还需要转化为 [N, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    # 加载存有具体分类的 json 信息并存储为字典
    json_path = './class_indices.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'file {json_path} does not exist')

    with open (json_path, 'r') as f:
        class_indict = json.load(f)    # 这里直接将 json 中的内容作为字典存储

    # print(class_indict)

    model = GoogLeNet(num_classes=5, aux_logits=False).to(device)

    weights_path = './googleNet.pth'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f'file {weights_path} does not exist')

    missing_keys, unexpeted_keys = model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

    model.eval()

    with torch.no_grad():
        # output = torch.squeeze(model(img.to(device))).cpu()
        # predict = torch.softmax(output, dim=0)
        # predict_cla = torch.argmax(predict).numpy()

        output1 = model(img.to(device))
        output2 = torch.squeeze(output1)
        predict = torch.softmax(output2, dim=-1)
        predict_cla = torch.argmax(predict).item()

    print(f'output1: {output1}')
    print(f'output2: {output2}')
    print_res = f'class: {class_indict[str(predict_cla)]}, prob: {predict[predict_cla].item():.3f}'

    plt.title(print_res)
    for i in range(len(predict)):
        print(f'class: {class_indict[str(i)]}    prob: {predict[i].item():.3f}')

    plt.show()

if __name__ == '__main__':
    main()