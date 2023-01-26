from utils.models import select_model
from torchvision import io
import torch
from torchvision import transforms
from random import sample
import os


def imageTransform(img):
    trans = transforms.Resize(img_size)
    img = trans(img)
    img = img.unsqueeze(0)
    img = img.type(torch.float32)
    return img


def getImage(path):
    img = io.read_image(path)
    return imageTransform(img)


def getIdxDict():
    return torch.load('idx2class')


def predictByTxt(net, txt, num_choice):
    net.eval()
    with open(txt) as f:
        datas = f.readlines()

    data_samples = sample(datas, num_choice)
    images = torch.tensor([])
    labels = []
    for data in data_samples:
        data = data.strip('\n')
        data = data.rstrip()
        X_path, label = data.split(' ')
        labels.append(label)
        image = io.read_image(X_path)
        image = torch.unsqueeze(image, dim=0)
        images = torch.cat([images, image.type(torch.float32)], dim=0)

    predicts = torch.argmax(net(torch.tensor(images)), dim=1)

    return images, [int(l) for l in labels], [int(p) for p in predicts]


def predictByImage(net, path):
    net.eval()
    image = io.read_image(path)
    image = torch.unsqueeze(image, dim=0)
    image = image.unsqueeze(dim=0)
    predict = torch.argmax(net(images), dim=1)
    return image, int(predict)


def predictByFile(net, path):
    net.eval()
    images = torch.tensor([])
    for name in os.listdir(path):
        image = io.read_image(os.path.join(path, name))
        image = torch.unsqueeze(image, dim=0)
        images = torch.cat([images, image.type(torch.float32)], dim=0)

    predicts = torch.argmax(net(torch.tensor(images)), dim=1)

    return images, [int(p) for p in predicts]


if __name__ == '__main__':
    in_channels = 1  # 输入通道数
    num_classes = 82  # 预测类别
    model_in_use = 'resNet18'  # 模型选用，可选项有：leNet, alexNet， vgg11, NiN, GoogLeNet, resNet18, denseNet
    model_kargs = {'in_channels': in_channels, "num_classes": num_classes}  # 模型参数，即输入通道数和总类别
    mode = ''  # image 为单个图片识别，file为从文件夹中识别，当他为空时从test.txt中时随机选择图片识别

    if model_in_use == 'leNet':
        img_size = (28, 28)
    elif model_in_use in ['alexNet', 'vgg11']:
        img_size = (224, 224)

    idxDict = getIdxDict()

    net = select_model(model_in_use, model_kargs)

    if mode == '':
        images, labels, predicts = predictByTxt(net, 'test.txt', 10)
        print([idxDict[l] for l in labels])
        print([idxDict[p] for p in predicts])

    elif mode == 'image':
        path = ""
        if path == "":
            print("路径不能为空")
        else:
            image, predict = predictByImage(net, path)
            print([idxDict[predict]])

    elif mode == "file":
        path = ""
        if path == "":
            print("路径不能为空")
        else:
            images, predicts = predictByFile(net, path)
            print([idxDict[p] for p in predicts])
