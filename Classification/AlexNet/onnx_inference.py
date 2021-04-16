# -*-coding: utf-8 -*-
import os
import sys

import torch
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.getcwd())


class ONNXModel(object):
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    @staticmethod
    def get_output_name(onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    @staticmethod
    def get_input_name(onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    @staticmethod
    def get_input_feed(input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy: tuple or list
        :return:
        """
        input_feed = {}
        if len(input_name) != len(image_numpy):
            raise ValueError('输入和模型实际输入个数不一致!')
        for i in range(len(input_name)):
            input_feed[input_name[i]] = image_numpy[i]
        return input_feed

    def forward(self, image_numpy):
        """
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy: tuple or list
        # :return:
        """
        # 输入数据的类型必须与模型一致
        # print(len(self.input_name))
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        outputs = self.onnx_session.run(self.output_name, input_feed=input_feed)
        result = {}
        for i in range(len(self.output_name)):
            result[self.output_name[i]] = outputs[i]
        return result


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    onnx_model_path = 'alexnet.onnx'
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
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    print(img.shape)
    print(img.dtype)

    yolov3Net = ONNXModel(onnx_model_path)
    out = yolov3Net.forward([to_numpy(img)])
    for i in range(len(yolov3Net.output_name)):
        print('==============================')
        print(out[yolov3Net.output_name[i]])