import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.pad = nn.ConstantPad2d(padding=(2, 2, 2, 2), value=0)

    def forward(self, x):
        result = self.pad(x)

        return result


if __name__ == '__main__':
    net = MyNet()
    inputs = torch.randn(1, 3, 224, 224)
    outputs = net(inputs)
    print(outputs.shape)

    # PyTorch转ONNX
    print('=================================')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 3, 352, 480).to(device)
    model = net.to(device)
    torch.onnx.export(
        model,
        dummy_input,
        'mynet.onnx',
        dynamic_axes={'image': {0: 'B'}, 'outputs': {0: 'B'}},
        input_names=['image'],
        output_names=['outputs'],
        opset_version=12,
        verbose=True,
        # do_constant_folding=True,
        # keep_initializers_as_inputs=True
    )
    print('=================================')

