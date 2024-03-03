import torch
import torch.nn as nn
from torch.nn import functional as F


class GlobalStochasticPooling(nn.Module):
    def __init__(self, num_segments=4):
        super().__init__()
        self.num_segments = num_segments

    def forward(self, x):
        b, c, h, w = x.shape
        # 计算概率，将所有零值替换为一个小的非零值，以避免除以零的情况
        nonzero_x = x + 1e-10
        p = F.adaptive_avg_pool2d(nonzero_x, 1) / nonzero_x.view(b, c, -1).sum(dim=2).reshape(b, c, 1, 1)
        # 修正概率，确保其满足概率的定义
        p /= p.sum(dim=(2, 3), keepdim=True)
        idx = torch.distributions.Multinomial(self.num_segments, p.squeeze()).sample().unsqueeze(2).unsqueeze(3)
        o = torch.zeros(b, c, 1, 1, device=x.device, dtype=x.dtype)  # 创建输出张量，大小为 1x1
        for i in range(self.num_segments):
            mask = (idx == i).float()
            segment = x * mask
            pooled_segment = F.adaptive_avg_pool2d(segment, (1, 1))  # 对部分x进行自适应平均池化，将其大小变为1x1
            o += pooled_segment
        return o

class MultiScaleChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(MultiScaleChannelAttentionModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sto_pool = GlobalStochasticPooling()

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, (1, 1), bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        sto_out = self.shared_MLP(self.sto_pool(x))
        return self.sigmoid(avg_out + max_out + sto_out)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7), stride=(1, 1), padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class MultiScaleSpatialAttentionModule(nn.Module):
    def __init__(self, channel):
        super(MultiScaleSpatialAttentionModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=(7, 7), stride=(1, 1), padding=3)
        self.sigmoid = nn.Sigmoid()
        self.conv0 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        avg_out1 = torch.mean(x1, dim=1, keepdim=True)
        max_out1, _ = torch.max(x1, dim=1, keepdim=True)

        avg_out2 = torch.mean(x2, dim=1, keepdim=True)
        max_out2, _ = torch.max(x2, dim=1, keepdim=True)

        avg_out3 = torch.mean(x3, dim=1, keepdim=True)
        max_out3, _ = torch.max(x3, dim=1, keepdim=True)

        out = torch.cat([avg_out1, max_out1, avg_out2, max_out2, avg_out3, max_out3], dim=1)
        out = self.sigmoid(self.conv0(out))

        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = MultiScaleChannelAttentionModule(channel)
        self.spatial_attention = MultiScaleSpatialAttentionModule(channel)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


#  建立瓶颈残差（组卷积）
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=(1, 1), downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=(1, 1),
                               stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups, kernel_size=(3, 3),
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# 建立残差网络 MARN
class MARN(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, groups=1, width_per_group=64):
        super(MARN, self).__init__()

        # 参数初始化
        self.in_channel = 64  # 输入通道数
        self.groups = groups  # 分组的数目
        self.width_per_group = width_per_group  # 分组后卷积核的深度

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.cbam_1 = CBAM(channel=256)

        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.cbam_2 = CBAM(channel=512)

        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.cbam_3 = CBAM(channel=1024)

        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.cbam_4 = CBAM(channel=2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channel, block_num, stride=(1, 1)):

        # 设置下采样的方式
        downsample = None

        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = [block(self.in_channel, channel, downsample=downsample, stride=stride,
                        groups=self.groups, width_per_group=self.width_per_group)]

        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam_1(x)
        x = self.layer2(x)
        x = self.cbam_2(x)

        x = self.layer3(x)
        x = self.cbam_3(x)
        x = self.layer4(x)
        x = self.cbam_4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
