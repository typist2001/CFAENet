import torch
import torch.nn as nn


# 通道减少 -> 通道注意力
class CCM(nn.Module):
    def __init__(self, infeature, out, redio):
        super(CCM, self).__init__()
        self.down = nn.Conv2d(infeature, out, kernel_size=1, stride=1)
        self.channel_attention = ChannelAttention(out, redio)

    def forward(self, x):
        x = self.down(x)
        w = self.channel_attention(x)
        return x * w


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 定义全局最大池化
        # 定义CBAM中的通道依赖关系学习层，注意这里是使用1x1的卷积实现的，而不是全连接层
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 实现全局平均池化
        max_out = self.fc(self.max_pool(x))  # 实现全局最大池化
        out = avg_out + max_out  # 两种信息融合
        # 最后利用sigmoid进行赋权
        return self.sigmoid(out)


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        # *号做实参 相当于解构  当形参表示一个可变长度的序列
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


class EdgeExtraction(nn.Module):
    def __init__(self, infeature):
        super(EdgeExtraction, self).__init__()
        self.conv = nn.Conv2d(infeature, infeature, kernel_size=1, stride=1)
        self.conv2 = DSConv3x3(infeature, infeature, stride=1, dilation=2)
        self.conv4 = DSConv3x3(infeature, infeature, stride=1, dilation=4)
        self.conv8 = DSConv3x3(infeature, infeature, stride=1, dilation=8)
        self.head = nn.Conv2d(2 * infeature, infeature, 7, 1, 3)

    def forward(self, x):
        y = self.conv(x)
        y2 = self.conv2(y)
        y4 = self.conv4(y)
        y8 = self.conv8(y)
        out = self.head(torch.cat([x, y2 + y4 + y8], dim=1))
        return out


class BFEM(nn.Module):
    def __init__(self):
        super(BFEM, self).__init__()

        self.conv_fuse2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=True),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(32, 1, kernel_size=3, padding=1)
                                        )

        self.conv_fuse1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
                                        nn.BatchNorm2d(128),
                                        nn.LeakyReLU(inplace=True)
                                        )

        self.conv_f1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(inplace=True)
                                     )
        self.conv_f2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(inplace=True)
                                     )

        self.dc1 = EdgeExtraction(128)

        self.ca1 = CCM(128, 64, redio=4)
        self.ca2 = CCM(128, 64, redio=4)

        self.shuffle1 = nn.PixelShuffle(2)

    # fuse 64 56 56     edge 28 28 128
    def forward(self, x, vgg112, vgg224):
        vgg112 = self.ca1(vgg112)
        q1 = self.conv_fuse1(
            torch.cat([x, vgg112], dim=1))
        edge112 = self.dc1(q1)
        q1 = self.shuffle1(self.conv_f1(torch.cat([q1, edge112], dim=1)))

        q2 = torch.cat([q1, vgg224], dim=1)
        q2 = self.ca2(self.conv_f2(q2))

        pre224 = self.conv_fuse2(q2)

        return pre224
