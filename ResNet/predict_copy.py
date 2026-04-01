# 这个是批量预测图片分类的代码
import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import resnet34

def main():
    computer = os.uname()[0]
    if computer == 'Darwin':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_root = './imgs'
    if not os.path.exists(img_root):
        raise FileNotFoundError(f'Image folder does not exist: {img_root}')
    img_path_list = [os.path.join(img_root, i) for i in os.listdir(img_root) if i.endswith('jpg') or i.endswith('jpeg')]
    # print(img_path_list)

    json_path = './class_indices.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'Json file does not exist: {json_path}')
    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    model = resnet34(num_classes=5).to(device)

    weight_path = './resNet34.pth'
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f'Weight file does not exist: {weight_path}')
    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval()
    batch_size = 2
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids+1) * batch_size]:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f'Image folder does not exist: {img_path}')
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            batch_img = torch.stack(img_list, dim=0)
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print(f'image: {img_path_list[ids * batch_size + idx]},  class: {class_indict[str(cla.numpy())]}, prob:{pro.numpy()*100:.3f}%')

if __name__ == '__main__':
    main()
