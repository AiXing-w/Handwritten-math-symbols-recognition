import torch
from torch import nn
from torch.nn import functional as F
import os

'''
leNet
'''


def leNet(in_channels=int, num_classes=int):
    net = nn.Sequential(nn.Conv2d(in_channels, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Flatten(),
                        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                        nn.Linear(120, 84), nn.Sigmoid(),
                        nn.Linear(84, num_classes)
                        )

    return net


'''
alexNet
'''


def alexNet(in_channels=int, num_classes=int):
    net = nn.Sequential(nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3, stride=2),
                  nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3, stride=2),
                  nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                  nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                  nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3, stride=2),
                  nn.Flatten(),
                  nn.Linear(6400, 4096), nn.ReLU(),
                  nn.Dropout(0.5),
                  nn.Linear(4096, 4096), nn.ReLU(),
                  nn.Dropout(0.5),
                  nn.Linear(4096, num_classes)
                  )

    return net


'''
vgg11
'''


def vgg_block(num_conv, in_channels, out_channels):
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg11(in_channel=int, num_classes=int):
    conv_archs = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    conv_blks = []
    for (num_conv, out_channel) in conv_archs:
        conv_blks.append(vgg_block(num_conv, in_channel, out_channel))
        in_channel = out_channel

    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channel * 7 * 7, 4096), nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 4096), nn.ReLU(),
                         nn.Linear(4096, num_classes)
                         )


'''
NiN
'''


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                         nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, kernel_size=1),
                         nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, kernel_size=1),
                         nn.ReLU()
                        )


def NiN(in_channels=int, num_classes=int):
    net = nn.Sequential(nin_block(in_channels, 96, 11, 4, 0),
                  nn.MaxPool2d(kernel_size=3, stride=2),
                  nin_block(96, 256, 5, 1, 2),
                  nn.MaxPool2d(kernel_size=3, stride=2),
                  nin_block(256, 384, 3, 1, 1),
                  nn.MaxPool2d(3, stride=2),
                  nn.Dropout(0.5),
                  nin_block(384, num_classes, 3, 1, 1),
                  nn.AdaptiveAvgPool2d((1, 1)),
                  nn.Flatten()
                  )
    return net


'''
GoogLeNet
'''


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 假设输入图像（1， 1， 96， 96）
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))

        return torch.cat((p1, p2, p3, p4), dim=1)


def GoogLeNet(in_channels, num_classes):
    b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)),
                       nn.Flatten()
                       )

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, num_classes))
    return net


'''
resNet18
'''


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)

        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)

        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))

    return blk


def resNet18(in_channels, num_classes):
    b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))

    b3 = nn.Sequential(*resnet_block(64, 128, 2))

    b4 = nn.Sequential(*resnet_block(128, 256, 2))

    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(512, num_classes)
                        )

    return net


'''
denseNet
'''


def conv_block(input_channels, num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(),
                         nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
                        )


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels
            ))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)

        return X


def transition_block(input_channels, num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(),
                         nn.Conv2d(input_channels, num_channels, kernel_size=1),
                         nn.AvgPool2d(kernel_size=2, stride=2)
                        )


def denseNet(in_channels, num_classes):
    b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        num_channels += num_convs * growth_rate
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    net = nn.Sequential(b1, *blks,
                        nn.BatchNorm2d(num_channels), nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(num_channels, num_classes)
                        )

    return net


def select_model(model_name=str, kwargs=dict):
    # 选择模型
    if model_name == 'leNet':
        net = leNet(kwargs['in_channels'], kwargs['num_classes'])

    elif model_name == 'alexNet':
        net = alexNet(kwargs['in_channels'], kwargs['num_classes'])

    elif model_name == 'vgg11':
        net = vgg11(kwargs['in_channels'], kwargs['num_classes'])

    elif model_name == 'NiN':
        net = NiN(kwargs['in_channels'], kwargs['num_classes'])

    elif model_name == 'GoogLeNet':
        net = GoogLeNet(kwargs['in_channels'], kwargs['num_classes'])

    elif model_name == 'resNet18':
        net = resNet18(kwargs['in_channels'], kwargs['num_classes'])

    elif model_name == 'denseNet':
        net = denseNet(kwargs['in_channels'], kwargs['num_classes'])

    if os.path.exists("./model_weights/{}.pth".format(model_name)):
        net.load_state_dict(torch.load("./model_weights/{}.pth".format(model_name)))
        print("model wieghts loaded")

    return net



