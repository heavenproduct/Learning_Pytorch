import torch
import torch.nn as nn

# 下载官方提供的预训练模型,
# 可以极大程度的减少训练时间(提升训练后的准确度)
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

# 定义网络类
# features 是指定某种具体的 vgg 名称
class VGG(nn.Module):
    def __init__(self,features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        # 卷积层提取特征(这里卷积层在别处定义, 根据选择的网络类型 vgg11,vgg13,vgg16,vgg19, 来调整)
        # 他们的设置里都是: kernel_size=3, padding=1, stride=1,
        # 所以输出的 特征图尺寸是不变的, 而且都是 5 个池化层
        # 所以特征图尺寸: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.features = features
        # 全连接层 进行分类 (上面提到的 vgg 网络在全连接层完全一致)
        self.classifier =  nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 卷积层提取特征
        # 输入: [N, 3, 224, 224]
        # 输出: [N, 512, 7, 7]
        x = self.features(x)

        # 展平操作：将特征图展平成特征向量
        # 输入: [N, 512, 7, 7]
        # 输出: [N, 512*7*7] = [N, 25088]
        # start_dim=1 表示从第1维开始展平，保留batch维度(N 就是第 0 维)
        x = torch.flatten(x, start_dim=1)  # 注意这里是 torch.flatten，不是 x.flatten

        # 全连接分类器
        # 输入: [N, 25088]
        # 输出: [N, num_classes]
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)

cfgs = {
    'vgg11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

# 根据调用时输入的 vgg 网络类型(如果能和 cfgs 中的对应上), 调整卷积层
def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

# 在训练中实际使用的函数, 调用 make_features 生成到具体的卷积层
# 然后调用 VGG, 实例化网络(所以在 train 中,不需要使用 net = VGG(###),
# 正常调用 vgg()就可以了)
def vgg(model_name='vgg16', **kwargs):
    assert model_name in cfgs, f"Warning: model number {model_name} not in cfgs dict!"
    # 这里 cfg 就是得到一个列表,model_name 就是字典的 key
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model