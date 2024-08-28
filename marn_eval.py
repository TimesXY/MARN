import xlwt
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from utils.marn import MARN, Bottleneck
from utils.utils_sr import roc_model, confusion, metrics_model
from utils.BroDataLoader import BronDataset

if __name__ == '__main__':
    # 添加到 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强
    img_size = 224
    valid_Transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()])

    # 加载数据集
    path = r'D://MyDataSet//BroCla01//'
    Data_test = BronDataset(path + "test.txt", transform=valid_Transform)

    # 划分数据集
    batch_size = 4
    data_test = DataLoader(Data_test, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 模型装配
    model = MARN(Bottleneck, [3, 4, 6, 3], num_classes=2, groups=32, width_per_group=4).to(device)

    # 加载模型
    model_weight_path = "save_weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 计算参数数目
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    # 模型保存
    model.eval()

    # 解决中文显示乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

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
    cf_matrix_colors = ['lightgray', 'steelblue']
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cf_matrix, annot=True, fmt='g', cmap=cf_matrix_colors, cbar=False, alpha=0.8)
    ax.set_title("Confusion Matrix of MARN", fontname='Times New Roman', fontsize=12)
    ax.set_xlabel("Predicted Class", fontproperties='Times New Roman', size=12)
    ax.set_ylabel("True Class", fontproperties='Times New Roman', size=12)
    plt.savefig('save_images//MARN_混淆矩阵.jpg', dpi=600)
    plt.show()

    # 测试集准确率
    metrics_model(model, data_test)

    """ 保存 roc 曲线结果用于重新绘制 """
    # 生成 Excel 设置Excel编码
    file = xlwt.Workbook('encoding = utf-8')

    # 创建sheet工作表
    sheet = file.add_sheet('sheet1', cell_overwrite_ok=True)

    # 先填标题
    sheet.write(0, 0, "FPR")
    sheet.write(0, 1, "TPR")

    # 循环填入数据
    for i in range(len(fpr_dict)):
        sheet.write(i + 1, 0, fpr_dict[i])
        sheet.write(i + 1, 1, tpr_dict[i])

    # 先填标题
    file.save('ROC_MARN.xls')
