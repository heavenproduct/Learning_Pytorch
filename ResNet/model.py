import torch
import torch.nn as nn


# BasicBlock 对应的是 18 层和 34 层的残差网络的网络的基本结构
# (既要拥有实线的残差结构 又要拥有虚线的残差结构(downsample 定义))
class BasicBlock(nn.Module):
    expansion = 1    # ResNet 18 和 34 的 conv2_x 到 conv5_x, 输入和输出的深度是一致的,所以用 1

    # 这里的 downsample 对应的是虚线部分的残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 这里的就是实线部分的结构, stride 就是默认值 1
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # --------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # --------------------------------------------------
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        # 这里相当于是将原始输入数据复制给 identity, 如果是实线的残差结构, 就直接和处理好的数据相加
        # 如果是虚线的残差结构, 就使用 downsample 定义的结构, 处理数据(H,W 变为原来的一半, 深度改变), 再相加
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # -----------------
        out = self.conv2(out)
        out = self.bn2(out)
        # -----------------
        out += identity
        out = self.relu(out)

        return out

# Bottleneck 对应的是 50 层, 101 层和 152 层的残差网络的网络的基本结构
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # --------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # --------------------------------------------------
        # 这里注意,因为第 3 层的深度是第 1 层和第 2 层的 4 倍, 所以要乘上 expansion
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=(out_channel * self.expansion), kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # --------------------------------------------------
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # -----------------
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # -----------------
        out = self.conv3(out)
        out = self.bn3(out)
        # -----------------
        out += identity
        out = self.relu(out)

        return out

# ----------------------------------------------------------------------------------------
# 正式定义 ResNet, 这里为了实现一个网络就能实现 浅层 ResNet (18, 34) 和深层 ResNet (50, 101, 152)
# 这里存在很多嵌套关系

# block_num 是一个列表参数,
# 例如: 对于 34 层网络, block_num=[3,4,6,3], 这里面的数字对应的就是con2_x, conv3_x, conv4_x, conv5_x中残差结构的数量
# include_top, 是为了能再 ResNet 的基础上修改出更加复杂的网络,这里暂时不使用
class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        # conv1 输入: shape=[batch, 3, 224, 224], 输出: shape=[batch, 64, 112, 112]
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -------------------------------------------------------------
        # 下面就是函数嵌套的,比较麻烦,但是深层和浅层的基本结构一致, 所以能统一用一套大模板
        # 下面定义的: _make_layer 生成具体的结构
        # conv2_x:
        # 对于 18 和 34 层, 输入: shape=[batch, 64, 112, 112], 输出: shape=[batch, 64, 56, 56], 全是实线结构
        # 对于 50, 101 和 152 层, 输入: shape=[batch, 64, 112, 112], 输出: shape=[batch, 256, 56, 56], 第一层是虚线结构(只调整深度)
        self.layer1 = self._make_layer(block, 64, block_num[0])
        # conv3_x:
        # 对于 18 和 34 层, 输入: shape=[batch, 64, 56, 56], 输出: shape=[batch, 128, 28, 28], 第一层是虚线结构(调整深度, H和W 减半)
        # 对于 50, 101 和 152 层, 输入: shape=[batch, 256, 56, 56], 输出: shape=[batch, 512, 28, 28], 第一层是虚线结构(调整深度, H和W 减半)
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        # conv4_x:
        # 对于 18 和 34 层, 输入: shape=[batch, 128, 28, 28], 输出: shape=[batch, 256, 14, 14], 第一层是虚线结构(调整深度, H和W 减半)
        # 对于 50, 101 和 152 层, 输入: shape=[batch, 512, 28, 28], 输出: shape=[batch, 1024, 14, 14], 第一层是虚线结构(调整深度, H和W 减半)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        # conv5_x:
        # 对于 18 和 34 层, 输入: shape=[batch, 256, 14, 14], 输出: shape=[batch, 512, 7, 7], 第一层是虚线结构(调整深度, H和W 减半)
        # 对于 50, 101 和 152 层, 输入: shape=[batch, 1024, 14, 14], 输出: shape=[batch, 2048, 7, 7],  第一层是虚线结构(调整深度, H和W 减半)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)

        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    # block 对应的是网络的基础结构(BasicBlock或者Bottleneck), 之后使用的 expansion 参数也是对应 block 中定义好的
    # 这里的 channel 对应的就是残差层第一层的 channel 的个数,
    # block_num 对应的是盖层包含了多少个残差结构, 是一个列表(同上)
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 这里是构建捷径部分(虚线结构中的左侧部分)
        # 对于 18, 34 层网络, 第一层(conv2_x)都会直接跳过这段语句,因为 stride=1(默认值), 同时block.expansion=1(这样就是全实线结构)
        # 对于 50, 101, 152层网络, 第一层(conv2_x)不会跳过这里, 因为block.expansion=4, 会生成一个虚线结构(但是 stride=1,所以只改变深度,不改变 H和W)
        # 对于18, 34, 50, 101, 152 层网络, conv3_x到 conv5_x, 都不会跳过这里, 会生成一个虚线结构 (因为 stride=2, 调整深度, H和W 减半)
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []

        # 这里构建的是残差结构的,第1层(实线和虚线结构都适用)
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        # 这里是构建残差结构后面层,都是实线结构
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)