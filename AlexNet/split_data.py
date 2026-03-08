import os
from shutil import copy
import random

def mkfile(file):
    """创建文件夹的函数，如果文件夹不存在则创建"""
    if not os.path.exists(file):
        os.makedirs(file)

# 原始图片文件夹路径
file_path = 'flower_data/flower_photos'

# 获取所有类别名（过滤掉.txt文件）-------只读取文件夹名
# flower_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla]
# # 结果：['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
# 为了适配 macOS, 排除自动创建的 .DS_Store文件
flower_class = [cla for cla in os.listdir(file_path) if os.path.isdir(file_path + '/' + cla) and not cla.startswith('.')]


# 创建训练集主文件夹和每个类别的子文件夹
mkfile('flower_data/train')
for cla in flower_class:
    mkfile('flower_data/train/' + cla)
    # 创建：train/daisy, train/dandelion, ...

# 创建验证集主文件夹和每个类别的子文件夹
mkfile('flower_data/val')
for cla in flower_class:
    mkfile('flower_data/val/' + cla)
    # 创建：val/daisy, val/dandelion, ...

split_rate = 0.1  # 10%作为验证集，90%作为训练集

# 对每个类别分别处理
for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'  # 当前类别的完整路径
    images = os.listdir(cla_path)  # 获取该类别所有图片的文件名列表

    num = len(images)  # 该类别图片总数

    # 随机选择10%的图片作为验证集
    eval_index = random.sample(images, k=int(num * split_rate))
    # 例如：如果该类别有100张图片，就随机选10张放入验证集

    # 遍历所有图片
    for index, image in enumerate(images):
        # 检查当前图片是否被选中为验证集
        if image in eval_index:
            # 是验证集图片
            image_path = cla_path + image  # 原始路径
            new_path = 'flower_data/val/' + cla  # 目标路径
            copy(image_path, new_path)  # 复制到验证集
        else:
            # 是训练集图片
            image_path = cla_path + image
            new_path = 'flower_data/train/' + cla
            copy(image_path, new_path)  # 复制到训练集

        # 打印进度条（实时显示处理进度）
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")

    print()  # 每个类别处理完后换行

print("processing done!")