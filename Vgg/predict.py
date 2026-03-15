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

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 首先定义好我们要检测的图片的地址
    img_path = "./蒲公英.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # 将图片传入
    img = Image.open(img_path)
    # 准备绘制图片
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)     # 对图片进行转换(裁切到 224*224)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)    # 形状: [1, 3, 224, 224] (增加batch维度)

    # 读取存储的 json 文件
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)    # class_indict 就是存储 每个种类的 key 和 value 的字典
    # {'0':'daisy', '1':'dandelion', '2':'roses', '3':'sunflower', '4':'tulips'}

    # 实例化模型
    model = vgg(model_name="vgg16", num_classes=5).to(device)
    # 加载模型训练好的参数
    weights_path = "./vgg16Net.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))    # 这里的 map_location 是确保权重加载到正确的设备

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()

        # model(img.to(device)): [1, 5] - 1张图片，5个类别得分
        # torch.squeeze(): [5] - 去掉batch维度
        # .cpu(): 移回CPU 这里必须移回 cpu, 后面的.numpy()才能顺利进行

        predict = torch.softmax(output, dim=0)
        # 将得分转化为 和为 1 的概率, predict = [0.05, 0.03, 0.89, 0.01, 0.02](这里的概率随便填的,反正是个列表)
        predict_cla = torch.argmax(predict).numpy()
        # print(predict_cla)

        # predict_cla = torch.argmax(predict).item()   # # 可以使用.item(), 只是为了将 Tensor 转换为 int
        # argmax: 找到概率最大的索引
        # .numpy(): 转为numpy数组
        # 所以这里的 predict_cla 是一个数字, 也就是 1

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()