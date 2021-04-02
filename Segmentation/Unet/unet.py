import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from vgg import create_vgg16


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # nn.UpsamplingBilinear2d trt不支持
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input1, input2):
        # print('input1.shape is: ', input1.shape)
        # print('input2.shape is: ', input2.shape)
        outputs = torch.cat([input1, self.up(input2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class UNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, pretrained=False):
        super(UNet, self).__init__()
        self.vgg = create_vgg16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64, 64, 512
        self.up_concat4 = UNetUp(in_filters[3], out_filters[3])
        # 128, 128, 256
        self.up_concat3 = UNetUp(in_filters[2], out_filters[2])
        # 256, 256, 128
        self.up_concat2 = UNetUp(in_filters[1], out_filters[1])
        # 512, 512, 64
        self.up_concat1 = UNetUp(in_filters[0], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        feat1 = self.vgg.features[0:4](inputs)
        feat2 = self.vgg.features[4:9](feat1)
        feat3 = self.vgg.features[9:16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)

        return final


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input1 = torch.randn(1, 3, 224, 224)
    model = UNet(pretrained=True)
    outputs = model(input1)
    print(outputs.shape)

    model.eval()

    # PyTorch转ONNX
    print('=================================')
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        'UNet.onnx',
        dynamic_axes={'image': {0: 'B'}, 'outputs': {0: 'B'}},
        input_names=['image'],
        output_names=['outputs'],
        opset_version=12,
        verbose=True
    )
    print('=================================')

    # PyTorch转TorchScript
    print('---------------------------------')
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save("UNet.pt")
    print('---------------------------------')


