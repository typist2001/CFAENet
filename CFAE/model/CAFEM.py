import torch
import torch.nn as nn
import math


class CAFEM(nn.Module):
    def __init__(self, infeature, size):
        super(CAFEM, self).__init__()
        self.infeature = infeature
        self.shape = size
        self.shuffle = nn.PixelShuffle(2)

        # 通道注意力
        self.ca = ChannelAttention(2 * self.infeature)
        # 空间注意力
        self.sa = SpatialAttention()

        self.cbr1 = conv3x3_bn_relu(2 * self.infeature, 2 * self.infeature)
        self.cbr2 = conv3x3_bn_relu(2 * self.infeature, 2 * self.infeature)
        self.cbr3 = conv3x3_bn_relu(2 * self.infeature, self.infeature)

        self.concatFuse = nn.Sequential(nn.Linear(2 * infeature, infeature),
                                        nn.GELU(),
                                        nn.Linear(infeature, infeature))
        self.att = Token_performer(dim=infeature, in_dim=infeature, kernel_ratio=0.5)

    def forward(self, rgb, depth, before):
        B, N, C = rgb.shape
        H, W = self.shape

        rgb_x = rgb.view(B, H, W, C).permute(0, 3, 1, 2)  # [b, 192, 56, 56]
        depth_x = depth.view(B, H, W, C).permute(0, 3, 1, 2)  # [b, 192, 56, 56]

        fuse = torch.cat([rgb_x, depth_x], dim=1)
        fuse = fuse * self.ca(fuse)
        fuse = fuse * self.sa(depth_x)

        fuse = self.cbr3(self.cbr2(self.cbr1(fuse)))

        fuse = fuse.flatten(2).transpose(1, 2)

        fuse = self.concatFuse(torch.cat([fuse, before], dim=2))
        fuse = self.att(fuse)

        return fuse


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 定义全局最大池化

        # 定义CBAM中的通道依赖关系学习层
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


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], 1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        super().__init__()
        self.emb = in_dim * head_cnt  # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        y = self.dp(self.proj(y))
        return y

    def forward(self, x):
        x = x + self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
