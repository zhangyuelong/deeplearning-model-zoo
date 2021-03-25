"""
Created by HenryMa on 2021/3/23
"""

__author__ = 'HenryMa'

from builtins import *
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_v3 import mobilenet_v3_large, mobilenet_v3_small


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load image
    img_path = 'a.jpg'
    assert os.path.exists(img_path), "file: '{}' does not exit.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    print(img.shape)

    # read class_indict
    json_path = 'class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    json_file = open(json_path, 'r')
    class_indict = json.load(json_file)
    # print(class_indict)

    # create model
    model = mobilenet_v3_small(num_classes=5).to(device)
    # load model weights
    model_weight_path = 'weights/MobileNetV3_Small.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    print('=================================')
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        'mobilenetv3_small.onnx',
        dynamic_axes={'image': {0: 'B'}, 'outputs': {0: 'B'}},
        input_names=['image'],
        output_names=['outputs'],
        opset_version=12
    )
    print('=================================')
    with torch.no_grad():
        # predict class
        import time
        t1 = time.time()
        model(img.to(device)).cpu()
        t2 = time.time()
        print('torch 推理花费{}ms'.format(1000*(t2-t1)))
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {} prob: {:.3}".format(class_indict[str(predict_cla)],
                                               predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
