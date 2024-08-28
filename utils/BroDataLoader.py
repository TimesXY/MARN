from PIL import Image
from torch.utils.data import Dataset


class BronDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        fh_txt = open(txt_path, 'r')
        images_labels = []
        for line in fh_txt:
            line = line.rstrip()  # 默认删除的是空白符 ('\n', '\r', '\t', ' ')
            words = line.split()  # 默认以空格、换行(\n)、制表符(\t)进行分割
            images_labels.append((words[0], int(words[1])))

        # 获取图像路径和标签
        self.images_labels = images_labels
        self.transform = transform

    def __getitem__(self, index):

        # 获取图像路径和标签，格式转换
        images_path, label = self.images_labels[index]
        images = Image.open(images_path).convert('RGB')

        # 数据增强
        if self.transform is not None:
            images = self.transform(images)

        return images, label

    def __len__(self):
        return len(self.images_labels)


class BronDatasetWithPath(Dataset):
    def __init__(self, txt_path, transform=None):
        fh_txt = open(txt_path, 'r')
        images_labels = []
        for line in fh_txt:
            line = line.rstrip()  # 默认删除的是空白符 ('\n', '\r', '\t', ' ')
            words = line.split()  # 默认以空格、换行(\n)、制表符(\t)进行分割
            images_labels.append((words[0], int(words[1])))

        # 获取图像路径和标签
        self.images_labels = images_labels
        self.transform = transform

    def __getitem__(self, index):

        # 获取图像路径和标签，格式转换
        images_path, label = self.images_labels[index]
        images = Image.open(images_path).convert('RGB')

        # 数据增强
        if self.transform is not None:
            images = self.transform(images)

        return images, label, images_path

    def __len__(self):
        return len(self.images_labels)
