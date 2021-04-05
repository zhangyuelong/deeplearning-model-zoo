import os
import argparse

import torch
from PIL import Image
import cv2
import numpy as np

from pspnet import PSPNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_num", type=int, default=2, help="预测的类别的种类")
    parser.add_argument("--weights", type=str, default="weights/PSPNet_weights.pth", help="训练好的权重路径")
    parser.add_argument("--colors", type=int, default=[[0, 0, 0], [0, 255, 0]], help="类别覆盖的颜色")
    parser.add_argument("--samples", type=str, default="samples/", help="用于测试的图片文件夹的路径")
    parser.add_argument("--outputs", type=str, default="outputs/", help="保存结果的文件夹的路径")
    args = parser.parse_args()
    print(args)

    CLASS_NUM = args.class_num
    WEIGHTS = args.weights
    COLORS = args.colors
    SAMPLES = args.samples
    OUTPUTS = args.outputs

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PSPNet(num_classes=CLASS_NUM, downsample_factor=16, pretrained=False, aux_branch=False).to(device=device)
    print('model structure is: ')
    print(model)

    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    # PyTorch转ONNX
    print('=================================')
    dummy_input = torch.randn(1, 3, 473, 473).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        'pspnet.onnx',
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
    traced_script_module.save("pspnet.pt")
    print('---------------------------------')

    paths = os.listdir(SAMPLES)

    for path in paths:
        image_src = cv2.imread(SAMPLES + path)
        image_shape = image_src.shape
        print(image_shape)
        image = cv2.resize(image_src, (473, 473))

        image = image / 255.0
        image = torch.Tensor(image).to(device)
        image = image.permute(2, 0, 1)
        image = torch.unsqueeze(image, dim=0)

        output = model(image)
        output = torch.squeeze(output)
        output = output.argmax(dim=0)
        output = output.cpu()
        output_np = cv2.resize(np.uint8(output), (image_shape[1], image_shape[0]))

        image_seg = np.zeros((image_shape[0], image_shape[1], 3))
        image_seg = np.uint8(image_seg)

        colors = COLORS

        for c in range(CLASS_NUM):
            image_seg[:, :, 0] += np.uint8((output_np == c)) * np.uint8(colors[c][0])
            image_seg[:, :, 1] += np.uint8((output_np == c)) * np.uint8(colors[c][1])
            image_seg[:, :, 2] += np.uint8((output_np == c)) * np.uint8(colors[c][2])

        image_seg = Image.fromarray(np.uint8(image_seg))
        old_image = Image.fromarray(np.uint8(image_src))

        image = Image.blend(old_image, image_seg, 0.3)

        # 将背景或空类去掉
        image_np = np.array(image)
        image_np[output_np == 0] = image_src[output_np == 0]
        image = Image.fromarray(image_np)
        image.save(OUTPUTS + path)

        print(path + " is done!")