import os
from shutil import copy
import random

# 定义一个创建文件夹的函数
def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)
        
# 指定要处理的文件夹路径
file_path = 'flower_data/flower_photos'

# 获取所有类别名（过滤掉.txt文件）-------只读取文件夹名
# 因为我只需要文件夹名字, 所以.txt一类的文件要过滤掉(macOS 的.DS_Store 文件也能处理掉)
flower_class = [cla for cla in os.listdir(file_path) if os.path.isdir(os.path.join(file_path+'/'+cla)) and not cla.startswith('.')]

# 创建训练集和验证集的文件夹以及内部对应的分类
# 先创建主文件夹在创建对应的子文件
mkfile('flower_data/train')
mkfile('flower_data/val')
for cla in flower_class:
    mkfile('flower_data/train' + '/' + cla)
    mkfile('flower_data/val' + '/' + cla)

# 确定划分比例
split_rata = 0.1

# 获取对应文件夹下的图片的名称并存入列表,然后复制到新的路径下
for cla in flower_class:
    # 先到目标目录下,获取存储的图片到列表
    cla_path = file_path + '/' + cla
    images = os.listdir(cla_path)

    # 统计图片总数
    num_images = len(images)

    # 将要划分到验证集中的图片名,单独存放到一个列表(随机抽取)
    eval_index = random.sample(images, k=int(split_rata * num_images))

    for index, image in enumerate(images):
        if image in eval_index:
            image_path = file_path + '/' + cla + '/' + image
            new_path = 'flower_data/val' + '/' + cla
            copy(image_path, new_path)

        else:
            image_path = file_path + '/' + cla + '/' + image
            new_path = 'flower_data/train' + '/' + cla
            copy(image_path, new_path)

        # 打印工作进度
        print(f'\r{cla} processing {index+1}/{num_images}', end='')

    print()  # 每个类别处理完后换行

print("processing done!")
