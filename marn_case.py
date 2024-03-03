import torch
import xlwt as xlwt

from torchvision import transforms
from utils.utils_sr import create_id
from torch.utils.data import DataLoader
from utils.resnext import ResNeXt, Bottleneck
from BronClassification.preprocessing.bro_dataloader import BronDatasetWithPath

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
    Data_test = BronDatasetWithPath(path + "test.txt", transform=valid_Transform)

    # 划分数据集
    batch_size = 4
    data_test = DataLoader(Data_test, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 模型装配
    model = ResNeXt(Bottleneck, [3, 4, 6, 3], num_classes=2, groups=32, width_per_group=4).to(device)

    # 加载模型
    model_weight_path = "save_weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 模型保存
    model.eval()

    # 生成表格数据
    save_path, save_real_label, save_predict_label, save_prob = create_id(model, data_test)

    # 生成 Excel 设置Excel编码
    file = xlwt.Workbook('encoding = utf-8')

    # 创建sheet工作表
    sheet = file.add_sheet('sheet1', cell_overwrite_ok=True)

    # 先填标题
    sheet.write(0, 0, "序号")
    sheet.write(0, 1, "真实类别")
    sheet.write(0, 2, "预测类别")
    sheet.write(0, 3, "类别0概率")
    sheet.write(0, 4, "类别1概率")

    # 循环填入数据
    for i in range(len(save_path)):
        sheet.write(i + 1, 0, save_path[i])
        sheet.write(i + 1, 1, save_real_label[i].item())
        sheet.write(i + 1, 2, save_predict_label[i].item())
        sheet.write(i + 1, 3, save_prob[i][0].item())
        sheet.write(i + 1, 4, save_prob[i][1].item())

    # 先填标题
    file.save('SaveData.xls')
