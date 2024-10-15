import numpy as np
from model.transformer_decoder import *
from model.VGG import LowFeatureExtract
from model.BFEM import BFEM


class BaseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, bias=True, norm_layer=False):
        super(BaseConv2d, self).__init__()
        self.basicconv = nn.Sequential()
        self.basicconv.add_module(
            'conv', nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias))
        if norm_layer:
            self.basicconv.add_module('bn', nn.BatchNorm2d(out_planes))
        self.basicconv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.basicconv(x)


class CFAENet(nn.Module):

    def __init__(self):
        super(CFAENet, self).__init__()
        pre_path = '/home/root803/gfq/pretrain/p2t_base.pth'
        from model.p2t import p2t_base
        self.backbone = p2t_base().cuda()
        self.depth_backbone = p2t_base().cuda()
        self.backbone.load_state_dict(torch.load(pre_path))
        self.depth_backbone.load_state_dict(torch.load(pre_path))

        self.low_feature_extract = LowFeatureExtract()
        self.transformer_decoder = TransformerDecoder_side()

        self.lin1 = nn.Linear(64, 96)
        self.lin2 = nn.Linear(128, 192)
        self.lin3 = nn.Linear(320, 384)
        self.lin4 = nn.Linear(512, 768)

        self.bfem = BFEM()

        self.conv_112 = BaseConv2d(96, 256)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, image, depth):
        b, c, h, w = image.size()
        depth = torch.cat([depth, depth, depth], dim=1)

        rgb1, rgb2, rgb3, rgb4 = self.backbone(image)  # list, length=5
        depth1, depth2, depth3, depth4 = self.depth_backbone(depth)

        out_7r = self.lin1(rgb1.flatten(2).transpose(1, 2))
        out_14r = self.lin2(rgb2.flatten(2).transpose(1, 2))
        out_28r = self.lin3(rgb3.flatten(2).transpose(1, 2))
        out_56r = self.lin4(rgb4.flatten(2).transpose(1, 2))

        outd_7r = self.lin1(depth1.flatten(2).transpose(1, 2))
        outd_14r = self.lin2(depth2.flatten(2).transpose(1, 2))
        outd_28r = self.lin3(depth3.flatten(2).transpose(1, 2))
        outd_56r = self.lin4(depth4.flatten(2).transpose(1, 2))

        rgb_features = out_7r, out_14r, out_28r, out_56r, out_56r
        depth_features = outd_7r, outd_14r, outd_28r, outd_56r, outd_56r
        # decoder
        x, sides = self.transformer_decoder(rgb_features, depth_features)  # [b, 3136, 96]
        x = x.view(b, 56, 56, 96).permute(0, 3, 1, 2)  # [b, 96, 56, 56]

        # BFEM
        feature_224, feature_112 = self.low_feature_extract(image)
        x = self.conv_112(x)
        x = self.shuffle(x)
        smap = self.bfem(x, feature_112, feature_224)

        return smap, sides  # , [e112, e224]


if __name__ == '__main__':
    rgb = np.random.random((1, 3, 224, 224))
    depth = np.random.random((1, 1, 224, 224))
    rgb = torch.Tensor(rgb).cuda()
    depth = torch.Tensor(depth).cuda()

    model = CFAENet()
    model.cuda()
    model.load_state_dict(torch.load("/home/root803/gfq/RGBD/CFAENet/checkpoints/CFAENet_best.pth"), False)
    model(rgb, depth)
