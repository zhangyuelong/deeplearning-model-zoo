import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data


bn_momentum = 0.1  # BN层的momentum
torch.cuda.manual_seed(1)  # 设置随机种子


# 编码器
class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

        self.encoder5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        indices = []

        x = self.encoder1(x)
        x, id1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)  # 保留最大值的位置
        indices.append(id1)
        x = self.encoder2(x)
        x, id2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        indices.append(id2)
        x = self.encoder3(x)
        x, id3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        indices.append(id3)
        x = self.encoder4(x)
        x, id4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        indices.append(id4)
        x = self.encoder5(x)
        x, id5 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        indices.append(id5)

        return x, indices


# 编码器 + 解码器
class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.weights_new = self.state_dict()
        self.encoder = Encoder(input_channels)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )

        self.decoder5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x, indices = self.encoder(x)

        # F.max_unpool2d转成onnx不支持, 用F.upsample替换
        # x = F.max_unpool2d(x, indices[4], kernel_size=(2, 2), stride=(2, 2))
        x = F.interpolate(x, scale_factor=(2, 2))
        x = self.decoder1(x)
        # x = F.max_unpool2d(x, indices[3], kernel_size=(2, 2), stride=(2, 2))
        x = F.interpolate(x, scale_factor=(2, 2))
        x = self.decoder2(x)
        # x = F.max_unpool2d(x, indices[2], kernel_size=(2, 2), stride=(2, 2))
        x = F.interpolate(x, scale_factor=(2, 2))
        x = self.decoder3(x)
        # x = F.max_unpool2d(x, indices[1], kernel_size=(2, 2), stride=(2, 2))
        x = F.interpolate(x, scale_factor=(2, 2))
        x = self.decoder4(x)
        # x = F.max_unpool2d(x, indices[0], kernel_size=(2, 2), stride=(2, 2))
        x = F.interpolate(x, scale_factor=(2, 2))
        x = self.decoder5(x)

        return x

    # 删掉VGG-16后面三个全连接层的权重
    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        del weights["classifier.0.weight"]
        del weights["classifier.0.bias"]
        del weights["classifier.3.weight"]
        del weights["classifier.3.bias"]
        del weights["classifier.6.weight"]
        del weights["classifier.6.bias"]

        names = []
        for key, value in self.encoder.state_dict().items():
            if "num_batches_tracked" in key:
                continue
            names.append(key)

        for name, dicts in zip(names, weights.items()):
            self.weights_new[name] = dicts[1]

        self.encoder.load_state_dict(self.weights_new)


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


if __name__ == "__main__":
    input_image = torch.randn(1, 3, 224, 224)
    enc = SegNet(input_channels=3, output_channels=128)
    out = enc(input_image)
    print(out.shape)
