import os
import torch
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import transforms
from collections import OrderedDict
from utils.utils_train import train
from torch.utils.data import DataLoader
from utils.marn import MARN, Bottleneck
from utils.utils_sr import roc_model, confusion, metrics_model
from utils.BroDataLoader import BronDataset

if __name__ == '__main__':
    # 添加到 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强
    img_size = 224
    train_Transform = transforms.Compose([transforms.Resize(img_size),
                                          transforms.CenterCrop(img_size),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.ToTensor()])

    valid_Transform = transforms.Compose([transforms.Resize(img_size),
                                          transforms.CenterCrop(img_size),
                                          transforms.ToTensor()])

    # 加载数据集
    path = r'D://MyDataSet//BroCla01//'
    Data_train = BronDataset(path + "train.txt", transform=train_Transform)
    Data_valid = BronDataset(path + "valid.txt", transform=valid_Transform)
    Data_test = BronDataset(path + "test.txt", transform=valid_Transform)

    # 划分数据集
    batch_size = 4
    data_train = DataLoader(Data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    data_valid = DataLoader(Data_valid, batch_size=batch_size, shuffle=False, pin_memory=True)
    data_test = DataLoader(Data_test, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 设置超参数
    epochs = 150
    weight_decay = 1e-4
    learning_rate = 2e-4

    # 模型装配
    model = MARN(Bottleneck, [3, 4, 6, 3], num_classes=2, groups=32, width_per_group=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    """ 加载预训练权重 """
    model_weight_path = "D:\\PycharmScript\\Models\\resnext50.pth"
    ckpt = torch.load(model_weight_path)
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    module_lst = [i for i in model.state_dict()]
    weights = OrderedDict()
    for idx, (k, v) in enumerate(ckpt.items()):
        if model.state_dict()[module_lst[idx]].shape == v.shape:
            weights[module_lst[idx]] = v
    model.load_state_dict(weights, strict=False)
    print('预训练权重加载完成')

    # 模型训练
    model, loss = train(model, data_train, data_valid, epochs=epochs, optimizer=optimizer)

    # 模型加载
    model_weight_path = "save_weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 解决中文显示乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制损失曲线
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(range(1, epochs + 1), loss['Loss1'], 'b-', label='train loss')
    ax1.plot(range(1, epochs + 1), loss['Loss2'], 'r-', label='validation loss')
    ax1.set_xlim(1, epochs)
    plt.xlabel("迭代次数", size=10)
    plt.ylabel("损失曲线", size=10)
    plt.legend(loc=1)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(1, epochs + 1), loss['Accuracy1'], 'b-', label='train accuracy')
    ax2.plot(range(1, epochs + 1), loss['Accuracy2'], 'r-', label='validation accuracy')
    ax2.set_ylim(0, 1)
    ax2.set_xlim(1, epochs)
    plt.xlabel("迭代次数", size=10)
    plt.ylabel("模型准确率", size=10)
    plt.legend(loc=1)

    plt.savefig('save_images//MARN_损失曲线.jpg', dpi=600)
    plt.show()

    # 绘制 ROC 曲线
    fpr_dict, tpr_dict, roc_dict = roc_model(model, data_test)

    plt.figure()
    plt.plot(fpr_dict, tpr_dict, label='ROC curve (area = {0:0.4f})'
                                       ''.format(roc_dict), color='r', linestyle='-.', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True  Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('save_images//MARN_ROC曲线.jpg', dpi=600)
    plt.show()

    # 绘制混淆矩阵
    cf_matrix = confusion(model, data_test)

    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')
    ax.title.set_text("Confusion Matrix")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    plt.savefig('save_images//MARN_混淆矩阵.jpg', dpi=600)
    plt.show()

    # 测试集准确率
    metrics_model(model, data_test)
