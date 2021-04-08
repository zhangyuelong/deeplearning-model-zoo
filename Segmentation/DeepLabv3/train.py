
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

from network.model import deeplabv3plus_resnet101, deeplabv3plus_resnet50, deeplabv3plus_mobilenet
from dataset import MyDataset


def main(class_num, pre_trained, train_data, batch_size, momentum, lr, cate_weight, epoch, weights):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = deeplabv3plus_mobilenet(num_classes=class_num, output_stride=8, pretrained_backbone=pre_trained)
    model.to(device)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(cate_weight)).float()).cuda()

    model.train()

    for i in range(epoch):
        for step, (b_x, b_y) in enumerate(train_loader):
            # print("b_x shape is: ", b_x.shape)
            # print('b_y shape is: ', b_y.shape)
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            b_y = b_y.view(-1, 352, 480)  # NHW
            output = model(b_x)
            loss = loss_func(output, b_y.long())  # nn.CrossEntropyLoss内部帮你做了max操作
            loss = loss.to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1 == 0:
                print("Epoch:{0} || Step:{1} || Loss:{2}".format(i, step, format(loss, ".4f")))

    torch.save(model.state_dict(), weights + "DeepLabV3plus_mobilenet_weights" + ".pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_num", type=int, default=2, help="训练的类别的种类")
    parser.add_argument("--epoch", type=int, default=20, help="训练迭代次数")
    parser.add_argument("--batch_size", type=int, default=4, help="批训练大小")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="学习率大小")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--category_weight", type=float, default=[0.7502381287857225, 1.4990483912788268],
                        help="损失函数中类别的权重")
    parser.add_argument("--train_txt", type=str, default="dataset/train.txt", help="训练的图片和标签的路径")
    parser.add_argument("--weights", type=str, default="./weights/", help="训练好的权重保存路径")
    parser.add_argument("--pre_trained", type=bool, default=True, help="是否使用预训练权重")
    args = parser.parse_args()
    print(args)

    CLASS_NUM = args.class_num
    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    MOMENTUM = args.momentum
    CATE_WEIGHT = args.category_weight
    TXT_PATH = args.train_txt
    PRE_TRAINED = args.pre_trained
    WEIGHTS = args.weights

    train_data = MyDataset(txt_path=TXT_PATH)

    main(CLASS_NUM, PRE_TRAINED, train_data, BATCH_SIZE, MOMENTUM, LR, CATE_WEIGHT, EPOCH, WEIGHTS)