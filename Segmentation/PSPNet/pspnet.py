from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mobilenetv2 import create_mobilenetv2


class NewMobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(NewMobileNetV2, self).__init__()
        model = create_mobilenetv2(pretrained=pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        # --------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
        # --------------------------------------------------------------------------------------------#
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    @staticmethod
    def _nostride_dilate(m, dilate):
        classname = m.__class__.__name__
        print(classname)
        if classname.find('Conv2d') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x_aux = self.features[:14](x)
        x = self.features[14:](x_aux)
        return x_aux, x


class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        #-----------------------------------------------------#
        #   分区域进行平均池化
        #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        #-----------------------------------------------------#
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer)
                                     for pool_size in pool_sizes])

        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    @staticmethod
    def _make_stages(in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='nearest')
                         for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))

        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, pretrained=True, aux_branch=True):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d

        #----------------------------------#
        #   获得两个特征层
        #   f4为辅助分支    [30,30,96]
        #   o为主干部分     [30,30,320]
        #----------------------------------#
        self.backbone = NewMobileNetV2(downsample_factor, pretrained)
        aux_channel = 96
        out_channel = 320

        # --------------------------------------------------------------#
        #	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   30,30,320 -> 30,30,80 -> 30,30,21
        # --------------------------------------------------------------#
        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(out_channel // 4, num_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch

        if self.aux_branch:
            #---------------------------------------------------#
            #	利用特征获得预测结果
            #   30, 30, 96 -> 30, 30, 40 -> 30, 30, 21
            #---------------------------------------------------#
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel//8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel//8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel//8, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.master_branch)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_aux, x = self.backbone(x)
        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='nearest')
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='nearest')
            return output_aux, output
        else:
            return output

    @staticmethod
    def initialize_weights(*models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()


if __name__ == '__main__':
    model = NewMobileNetV2(downsample_factor=16, pretrained=True)
    inputs = torch.randn(1, 3, 473, 473)
    # 输出size是1, 96, 30, 30和1, 320, 30, 30
    outputs = model(inputs)
    print(outputs[0].shape, outputs[1].shape)
    inputs = torch.randn(4, 320, 30, 30)
    PSP = _PSPModule(in_channels=320, pool_sizes=[1, 2, 3, 6], norm_layer=nn.BatchNorm2d)
    PSP.eval()
    output = PSP(inputs)
    print(output.shape)
    inputs = torch.randn(4, 3, 473, 473)
    psp = PSPNet(num_classes=2, downsample_factor=16, pretrained=True, aux_branch=False)
    psp.eval()
    output = psp(inputs)
    print(output.shape)
