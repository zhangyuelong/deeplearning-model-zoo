import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.models.utils import load_state_dict_from_url


def create_vgg16_bn(pretrained=False, model_dir='./weights'):
    backbone_net = models.vgg16_bn(pretrained=False)
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
                                              model_dir='./weights')
        pre_dict = {k: v for k, v in state_dict.items() if backbone_net.state_dict()[k].numel() == v.numel()}
        backbone_net.load_state_dict(pre_dict, strict=False)

    return backbone_net


class FCN(nn.Module):
    def __init__(self, num_classes, backbone='vgg16_bn', pretrained=True):
        super(FCN, self).__init__()

        if backbone == 'vgg16_bn':
            backbone_net = create_vgg16_bn(pretrained=pretrained)
            self.stage1 = backbone_net.features[:7]
            self.stage2 = backbone_net.features[7:14]
            self.stage3 = backbone_net.features[14:24]
            self.stage4 = backbone_net.features[24:34]
            self.stage5 = backbone_net.features[34:]

            self.scores1 = nn.Conv2d(512, num_classes, 1)
            self.scores2 = nn.Conv2d(512, num_classes, 1)
            self.scores3 = nn.Conv2d(128, num_classes, 1)

            self.conv_trans1 = nn.Conv2d(512, 256, 1)
            self.conv_trans2 = nn.Conv2d(256, num_classes, 1)

            # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, **args)
            self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
            self.upsample_8x.weight.data = self.bilinear_kernel(num_classes, num_classes, 16)

            self.upsample_2x_1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
            self.upsample_2x_1.weight.data = self.bilinear_kernel(512, 512, 4)

            self.upsample_2x_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
            self.upsample_2x_2.weight.data = self.bilinear_kernel(256, 256, 4)

        else:
            raise NotImplementedError('????????????backbone{}'.format(backbone))

    @staticmethod
    def bilinear_kernel(in_channels, out_channels, kernel_size):
        """
        Define a bilinear kernel according to in_channels and out_channels.
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :return: return a bilinear tensor
        """
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
        weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
        return torch.from_numpy(weight)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)

        scores1 = self.scores1(s5)
        s5 = self.upsample_2x_1(s5)
        add1 = s5 + s4

        scores2 = self.scores2(add1)

        add1 = self.conv_trans1(add1)
        add1 = self.upsample_2x_2(add1)
        add2 = add1 + s3

        output = self.conv_trans2(add2)
        output = self.upsample_8x(output)

        return output


if __name__ == '__main__':
    vgg16_bn = create_vgg16_bn(pretrained=True)
    print(vgg16_bn.features)
    fcn = FCN(2, backbone='vgg16_bn')
    inputs = torch.randn(1, 3, 352, 480)
    outputs = fcn(inputs)
    print(outputs.shape)







