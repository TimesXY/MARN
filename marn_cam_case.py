import torch
import numpy as np

from PIL import Image
from utils.marn import MARN, Bottleneck
from torchvision import transforms
from utils.utils_cam import GradCAM, show_cam_on_image

if __name__ == '__main__':
    """
    可调整参数
    target_layers    : 可视化目标层
    model            : 实例化对应模型
    target_category  : 选择的对应类别
    img_path         : 图像的对应路径
    model_weight_path: 模型的权重路径
    """

    # 设置路径
    target_category = 1
    img_path = "D:\\MyDataSet\\BroCla01\\1\\"
    model_weight_path = "save_weights/best_model.pth"

    # 建立模型
    model = MARN(Bottleneck, [3, 4, 6, 3], num_classes=2, groups=32, width_per_group=4)
    target_layers = [model.layer4[2]]

    # 加载模型
    ckpt = torch.load(model_weight_path)
    model.load_state_dict(ckpt, strict=False)
    print("预训练加载完成")

    # 图像处理
    img_size = 224
    Transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(img_size),
                                    transforms.CenterCrop(img_size)])

    # CAM 实例化
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # 循环保存可视化图
    file_name = '411005.jpg'
    images_path = img_path + file_name

    # 获取原始图像
    img = Image.open(images_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    # 原始图像的格式转换
    img_tensor = Transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # CAM 结果图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]

    # CAM 和原始图像叠加显示
    img_tensor = img_tensor.permute(1, 2, 0).numpy()
    visualization = show_cam_on_image(img_tensor, grayscale_cam, use_rgb=True)
    visualization = Image.fromarray(np.uint8(visualization))
    visualization.save("save_images\\images_show\\" + file_name)
