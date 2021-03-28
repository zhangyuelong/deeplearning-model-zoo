import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

from SegNet import SegNet, MyDataset


def main(class_num, pre_trained, train_data, batch_size, momentum, lr, cate_weight, epoch, weights):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SegNet(3, class_num)
    model = model.to(device)
    model.load_weights(pre_trained)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(cate_weight)).float()).cuda()

    model.train()
    for i in range(epoch):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            b_y = b_y.view(-1, 224, 224)
            output = model(b_x)
            loss = loss_func(output, b_y.long())
            loss = loss.to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1 == 0:
                print("Epoch:{0} || Step:{1} || Loss:{2}".format(i, step, format(loss, ".4f")))

    torch.save(model.state_dict(), weights + "SegNet_weights" + ".pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_num", type=int, default=2, help="训练的类别的种类")
    parser.add_argument("--epoch", type=int, default=20, help="训练迭代次数")
    parser.add_argument("--batch_size", type=int, default=8, help="批训练大小")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="学习率大小")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--category_weight", type=float, default=[0.7502381287857225, 1.4990483912788268],
                        help="损失函数中类别的权重")
    parser.add_argument("--train_txt", type=str, default="dataset/train.txt", help="训练的图片和标签的路径")
    parser.add_argument("--pre_trained_weight", type=str, default="weights/vgg16_bn-6c64b313.pth", help="编码器预训练权重路径")
    parser.add_argument("--weights", type=str, default="./weights/", help="训练好的权重保存路径")
    args = parser.parse_args()
    print(args)

    class_num = args.class_num
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.learning_rate
    momentum = args.momentum
    cate_weight = args.category_weight
    txt_path = args.train_txt
    pre_trained = args.pre_trained_weight
    weights = args.weights

    train_data = MyDataset(txt_path=txt_path)

    main(class_num, pre_trained, train_data, batch_size, momentum, lr, cate_weight, epoch, weights)



