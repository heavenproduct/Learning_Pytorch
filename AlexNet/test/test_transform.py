# 本代码是为了实验 transform 里面一些对于图像变换的函数的效果
# transform 只能接受 PIL 或者 numpy 的数据类型, Tensor 要转换为 numpy
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt


def load_image(image):
    img = Image.open(image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


image_path = "RMB.jpg"
# 未经处理的图片
the_test_img = load_image(image_path)

# 定义变换类型
# transform = transforms.RandomResizedCrop(224)           # 这个是先随机裁剪,再缩放成 224*224
# transform = transforms.RandomResizedCrop([224, 112])    # 这个是先随机裁剪,再缩放成 224*112
# transform = transforms.RandomHorizontalFlip(p=0.9)      # 水平方向随机翻转(镜像)，概率为 0.9, 即一半的概率翻转, 一半的概率不翻转
# transform = transforms.Resize((112,112))                # 图像调整为112*112像素(正方形)
# transform = transforms.Resize(112)                        # 图像的较短边调整为112像素，而较长边将按比例缩放
transform = transforms.RandomRotation(45, center=(0, 0))

# 处理过的图像
transformed_img = transform(the_test_img)

# 展示处理前和处理后
plt.figure(figsize=(10,5))

# 定义左边为原图
plt.subplot(1, 2, 1)    # 1行2列，第1个位置
plt.imshow(the_test_img)
plt.title("Original Image")
plt.axis("off")               # 不显示坐标轴
# 定义右边为变换后的图片
plt.subplot(1, 2, 2)    # 1行2列，第2个位置
plt.imshow(transformed_img)
plt.title("Transformed Image")
plt.axis("off")

plt.tight_layout()            # 自动布局
plt.show()