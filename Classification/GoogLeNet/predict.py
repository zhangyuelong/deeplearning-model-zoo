import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import GoogLeNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "./a.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = GoogLeNet(num_classes=5, aux_logits=False).to(device)

    # load model weights
    weights_path = "./weights/googlenet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                          strict=False)

    model.eval()
    print('=================================')
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        'googlenet.onnx',
        dynamic_axes={'image': {0: 'B'}, 'outputs': {0: 'B'}},
        input_names=['image'],
        output_names=['outputs'],
        opset_version=12
    )
    print('=================================')

    print('---------------------------------')
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save("googlenet.pt")
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()