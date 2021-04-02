import cv2
import torch
import torch.utils.data as Data


class MyDataset(Data.Dataset):
    def __init__(self, txt_path):
        super(MyDataset, self).__init__()

        paths = open(txt_path, "r")

        image_label = []
        for line in paths:
            line = line.rstrip("\n")
            line = line.lstrip("\n")
            path = line.split(';')
            image_label.append((path[0], path[1]))

        self.image_label = image_label

    def __getitem__(self, item):
        image, label = self.image_label[item]

        image = 'dataset/jpg/' + image
        label = 'dataset/png/' + label

        image = cv2.imread(image)
        image = cv2.resize(image, (224, 224))
        image = image/255.0  # 归一化输入
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)  # 将图片的维度转换成网络输入的维度（channel, width, height）

        label = cv2.imread(label, 0)
        label = cv2.resize(label, (224, 224))
        label = torch.Tensor(label)

        return image, label

    def __len__(self):
        return len(self.image_label)
