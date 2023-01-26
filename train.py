import torch
from torch import nn
from tqdm import tqdm
from utils.dataLoader import LoadDataset
from utils.models import select_model
from utils.plotShow import plot_history


def accuracy(y_hat, y):
    # 预测精度
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def train(model_in_use, model_kargs, train_iter, test_iter, num_epochs, lr, device, threshold, save_checkpoint=False):
    # 训练模型
    net = select_model(model_in_use, model_kargs)

    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print("device in : ", device)
    net = net.to(device)

    loss = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        data_num = 0

        with tqdm(range(len(train_iter)), ncols=100, colour='red',
                  desc="{} train epoch {}/{}".format(model_in_use, epoch + 1, num_epochs)) as pbar:
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                train_loss += l.detach()
                train_acc += accuracy(y_hat.detach(), y.detach())
                data_num += X.shape[0]
                pbar.set_postfix({'loss': "{:.4f}".format(train_loss / data_num), 'acc': "{:.4f}".format(train_acc / data_num)})
                pbar.update(1)

        history['train_loss'].append(float(train_loss / data_num))
        history['train_acc'].append(float(train_acc / data_num))

        net.eval()
        test_loss = 0.0
        test_acc = 0.0
        data_num = 0
        with tqdm(range(len(test_iter)), ncols=100, colour='blue',
                  desc="{} test epoch {}/{}".format(model_in_use, epoch + 1, num_epochs)) as pbar:
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                with torch.no_grad():
                    l = loss(y_hat, y)
                    test_loss += l.detach()
                    test_acc += accuracy(y_hat.detach(), y.detach())

                    data_num += X.shape[0]
                    pbar.set_postfix({'loss': "{:.4f}".format(test_loss / data_num), 'acc': "{:.4f}".format(test_acc / data_num)})
                    pbar.update(1)

        history['test_loss'].append(float(test_loss / data_num))
        history['test_acc'].append(float(test_acc / data_num))
        if history['test_acc'][-1] > threshold:
            print("early stop")
            break
        if save_checkpoint:
            torch.save(net.state_dict(), "./model_weights/ep{}-{}-acc-{:.4f}-loss-{:.4f}.pth".format(epoch+1, model_in_use, history['test_acc'][-1], history['test_loss'][-1]))

    torch.save(net.state_dict(), "./model_weights/{}.pth".format(model_in_use))
    return history


if __name__ == '__main__':
    batch_size = 32  # 批量大小
    in_channels = 1  # 输入通道数
    num_classes = 82  # 预测类别
    num_epochs = 5  # 训练轮次
    lr = 5e-3
    threshold = 0.95  # 提前停止的阈值，即测试精度超过这个阈值就停止训练
    data_path = r"./extracted_images"  # 数据路径
    model_in_use = 'leNet'  # 模型选用，可选项有：leNet, alexNet， vgg11, NiN, GoogLeNet, resNet18, denseNet
    model_kargs = {'in_channels': in_channels, "num_classes": num_classes}  # 模型参数，即输入通道数和总类别
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 测试cuda并使用
    img_size = None
    if model_in_use == 'leNet':
        img_size = (28, 28)
    elif model_in_use in ['alexNet', 'vgg11']:
        img_size = (224, 224)

    train_iter, test_iter = LoadDataset(data_path=data_path, batch_size=batch_size, img_size=img_size)  # 加载数据集

    history = train(model_in_use, model_kargs, train_iter, test_iter, num_epochs, lr, device, threshold)  # 训练

    plot_history(model_in_use, history)  # 画出损失和精度的图
