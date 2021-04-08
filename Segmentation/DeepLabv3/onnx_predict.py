import os

import torch
from PIL import Image
import cv2
import numpy as np
import onnxruntime


if __name__ == '__main__':
    SAMPLES = './samples/'
    OUTPUTS = './outputs/'
    COLORS = [[0, 0, 0], [0, 255, 0]]
    CLASS_NUM = 2

    session = onnxruntime.InferenceSession('deeplabv3+_mobilenet_opt.onnx')
    first_input_name = session.get_inputs()[0].name
    first_output_name = session.get_outputs()[0].name

    paths = os.listdir(SAMPLES)
    print(paths)

    for path in paths:
        image_src = cv2.imread(SAMPLES + path)
        image_shape = image_src.shape
        print(image_shape)
        image = cv2.resize(image_src, (480, 352))

        image = image / 255.0
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        image = torch.unsqueeze(image, dim=0)

        image = image.numpy()

        #==========================onnx推理==============================#
        results = session.run([first_output_name], {first_input_name: image})
        output = results[0]
        output = torch.from_numpy(output)
        #==========================onnx推理==============================#
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