import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import vgg

def main():
    # 使用 GPU 训练, 自动判断是 Mac的显卡 或者 NVIDIA 显卡
    computer = os.uname()
    # print(computer[0])
    if computer[0] == 'Darwin':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 载入需要判断的图片
    img_path = './蒲公英.jpg'
    if not os.path.exists(img_path):
        raise FileNotFoundError
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError
    with open(json_path, 'r') as f:
        class_indicts = json.load(f)

    model = vgg(model_name='vgg16', num_classes=5)
    model.to(device)

    weights_path = './vgg16Net.pth'
    if not os.path.exists(weights_path):
        raise FileNotFoundError
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        img = img.to(device)
        output = torch.squeeze(model(img)).cpu()
        predict = torch.softmax(output, dim=-1)
        predict_cla = torch.argmax(predict).numpy()

    print_res = f'class:{class_indicts[str(predict_cla)]}, prob:{predict[predict_cla].numpy():.3f}'
    plt.title(print_res)
    plt.show()

if __name__ == '__main__':
    main()