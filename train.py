import time
import matplotlib.pyplot as plt
import torch.utils.data
import torch
from torch import nn
from torch import optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.nn import functional as F
import numpy as np
from thop import profile
from MLP import MLP
from CNN import CNN
from ResNet import ResNet
from vision_transformer import ViT
from VIT_CNN import ViT_CNN
from VIT_ResNet import ViT_ResNet


BATCH_SIZE = 128
TEST_BATCH_SIZE = 1024
EPOCH_SIZE = 30


def get_dataloader(train=True, batch_size=BATCH_SIZE):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    dataset = FashionMNIST(root='./data', train=train, download=True, transform=transform_fn)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train(model, optimizer):
    mode = True
    model.train(mode=mode)

    train_dataloader = get_dataloader(train=True, batch_size=BATCH_SIZE)
    for idx, (data, target) in enumerate(train_dataloader):
        data = data.to('cuda')
        target = target.to('cuda')
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model):
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=False, batch_size=TEST_BATCH_SIZE)
    for idx, (input, target) in enumerate(test_dataloader):
        input = input.to('cuda')
        target = target.to('cuda')
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss.cpu())
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu())
    print("平均准确率：{},平均损失：{}".format(np.mean(acc_list), np.mean(loss_list)))
    return np.mean(acc_list)


def val(model):
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=True, batch_size=TEST_BATCH_SIZE)
    for idx, (input, target) in enumerate(test_dataloader):
        input = input.to('cuda')
        target = target.to('cuda')
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss.cpu())
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu())
    print("平均准确率：{},平均损失：{}".format(np.mean(acc_list), np.mean(loss_list)))
    return np.mean(acc_list)


def draw_plot(val_pred, test_pred, name):
    x = []
    for i in range(EPOCH_SIZE):
      x.append(i)
    y1 = val_pred
    y2 = test_pred
    # 绘制折线图，添加数据点，设置点的大小
    # * 表示绘制五角星；此处也可以不设置线条颜色，matplotlib会自动为线条添加不同的颜色
    plt.plot(x, y1, 'r', marker='*', markersize=2)
    plt.plot(x, y2, 'b', marker='*', markersize=2)
    plt.title('')  # 折线图标题
    plt.xlabel('epoch')  # x轴标题
    plt.ylabel('acc')  # y轴标题
    # 给图像添加注释，并设置样式
    for a, b in zip(x, y1):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=2)
    for a, b in zip(x, y2):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=2)
    # 绘制图例
    plt.legend(['val', 'test'])
    # 显示图像
    plt.savefig('./' + name + '.png',  # ⽂件名：png、jpg、pdf
                dpi=100,  # 保存图⽚像素密度
                facecolor='violet',  # 视图与边界之间颜⾊设置
                edgecolor='lightgreen',  # 视图边界颜⾊设置
                bbox_inches='tight')  # 保存图⽚完整
    plt.clf()


def draw_plot2(val_pred, test_pred, name):
    y1 = val_pred
    y2 = test_pred
    txt = ['MLP','CNN','','ViT','ViT+CNN','']
    # 绘制折线图，添加数据点，设置点的大小
    # * 表示绘制五角星；此处也可以不设置线条颜色，matplotlib会自动为线条添加不同的颜色
    plt.scatter(y1, y2)
    plt.title('')  # 折线图标题
    plt.xlabel('GFLOPs')  # x轴标题
    plt.ylabel('params')  # y轴标题
    for i in range(6):
      plt.annotate(txt[i], xy = (y1[i], y2[i]), xytext = (y1[i]+0.1, y2[i]+0.1))
    # 给图像添加注释，并设置样式
    # 绘制图例
    # plt.legend(['val', 'test'])
    # 显示图像
    plt.savefig('./' + name + '.png',  # ⽂件名：png、jpg、pdf
                dpi=100,  # 保存图⽚像素密度
                facecolor='violet',  # 视图与边界之间颜⾊设置
                edgecolor='lightgreen',  # 视图边界颜⾊设置
                bbox_inches='tight')  # 保存图⽚完整
    plt.clf()

MLP = MLP().to('cuda')
optimizer_MLP = optim.Adam(MLP.parameters(), lr=0.001)
CNN = CNN().to('cuda')
optimizer_CNN = optim.Adam(CNN.parameters(), lr=0.001)
ResNet = ResNet().to('cuda')
optimizer_Res = optim.Adam(ResNet.parameters(), lr=0.001)
ViT = ViT(
    image_size=28,
    patch_size=7,
    num_classes=10,
    dim=49,
    depth=4,
    heads=7,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
    channels=1,
    dim_head=7
).to('cuda')
optimizer_ViT = optim.Adam(ViT.parameters(), lr=0.001)
ViT_CNN = ViT_CNN(
    image_size=28,
    patch_size=7,
    num_classes=10,
    dim=49,
    depth=4,
    heads=7,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
    channels=1,
    dim_head=7
).to('cuda')
optimizer_ViT_CNN = optim.Adam(ViT_CNN.parameters(), lr=0.001)
ViT_Res = ViT_ResNet(
    image_size=28,
    patch_size=7,
    num_classes=10,
    dim=49,
    depth=4,
    heads=7,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
    channels=1,
    dim_head=7
).to('cuda')
optimizer_ViT_Res = optim.Adam(ViT_Res.parameters(), lr=0.001)
if __name__ == '__main__':
    input = torch.randn(1,1,28,28).to('cuda')
    GFLOPs = []
    params = []
    mac, param = profile(MLP,inputs=(input,))
    GFLOPs.append(mac)
    params.append(param)
    mac, param = profile(CNN,inputs=(input,))
    GFLOPs.append(mac)
    params.append(param)
    mac, param = profile(ResNet,inputs=(input,))
    GFLOPs.append(mac)
    params.append(param)
    mac, param = profile(ViT,inputs=(input,))
    GFLOPs.append(mac)
    params.append(param)
    mac, param = profile(ViT_CNN,inputs=(input,))
    GFLOPs.append(mac)
    params.append(param)
    mac, param = profile(ViT_Res,inputs=(input,))
    GFLOPs.append(mac)
    params.append(param)
    draw_plot2(GFLOPs, params, "GFLOPs+Params")
    epoch = EPOCH_SIZE
    val_pred = []
    test_pred = []
    for i in range(epoch):
        train(ResNet, optimizer_Res)
        pred = val(ResNet)
        val_pred.append(pred)
        pred = test(ResNet)
        test_pred.append(pred)
    draw_plot(val_pred, test_pred, "ResNet")
    val_pred = []
    test_pred = []
    for i in range(epoch):
        train(MLP, optimizer_MLP)
        pred = val(MLP)
        val_pred.append(pred)
        pred = test(MLP)
        test_pred.append(pred)
    draw_plot(val_pred, test_pred, "MLP")
    val_pred = []
    test_pred = []
    for i in range(epoch):
        train(CNN, optimizer_CNN)
        pred = val(CNN)
        val_pred.append(pred)
        pred = test(CNN)
        test_pred.append(pred)
    draw_plot(val_pred, test_pred, "CNN")
    val_pred = []
    test_pred = []
    for i in range(epoch):
        train(ViT, optimizer_ViT)
        pred = val(ViT)
        val_pred.append(pred)
        pred = test(ViT)
        test_pred.append(pred)
    draw_plot(val_pred, test_pred, "ViT")
    val_pred = []
    test_pred = []
    for i in range(epoch):
        train(ViT_CNN, optimizer_ViT_CNN)
        pred = val(ViT_CNN)
        val_pred.append(pred)
        pred = test(ViT_CNN)
        test_pred.append(pred)
    draw_plot(val_pred, test_pred, "ViT_CNN")
    val_pred = []
    test_pred = []
    for i in range(epoch):
        train(ViT_Res, optimizer_ViT_Res)
        pred = val(ViT_Res)
        val_pred.append(pred)
        pred = test(ViT_Res)
        test_pred.append(pred)
    draw_plot(val_pred, test_pred, "Vit_Res")
    val_pred = []
    test_pred = []
