import os
import numpy as np

if __name__ == '__main__':
    # 设置数据集路径
    data_root = r"D:\MyDataSet\BI2K"

    # 获取类别文件夹（0 和 1）
    main_dirs = ['0', '1']
    subdir_files = {}

    # 遍历类别文件夹，收集每个子文件夹中的图片路径
    for main_dir in main_dirs:
        main_dir_path = os.path.join(data_root, main_dir)
        if os.path.isdir(main_dir_path):
            sub_dirs = os.listdir(main_dir_path)
            for sub_dir in sub_dirs:
                sub_dir_path = os.path.join(main_dir_path, sub_dir)
                if os.path.isdir(sub_dir_path):
                    img_names = os.listdir(sub_dir_path)
                    file_list = [os.path.join(sub_dir_path, img_name) + '\t' + main_dir for img_name in img_names]
                    subdir_files[sub_dir_path] = file_list

    # 将子文件夹路径打乱
    subdir_paths = list(subdir_files.keys())
    np.random.shuffle(subdir_paths)

    # 按比例划分训练集、验证集和测试集
    num_subdirs = len(subdir_paths)
    train_cutoff = int(num_subdirs * 0.6)
    valid_cutoff = int(num_subdirs * 0.8)

    train = []
    valid = []
    test = []

    for i, subdir_path in enumerate(subdir_paths):
        if i < train_cutoff:
            train.extend(subdir_files[subdir_path])
        elif i < valid_cutoff:
            valid.extend(subdir_files[subdir_path])
        else:
            test.extend(subdir_files[subdir_path])

    # 建立 txt 文件用于存储序号
    with open(os.path.join(data_root, 'train.txt'), 'w') as f_train, \
            open(os.path.join(data_root, 'valid.txt'), 'w') as f_valid, \
            open(os.path.join(data_root, 'test.txt'), 'w') as f_test:

        for item in train:
            f_train.write(item + '\n')
        for item in valid:
            f_valid.write(item + '\n')
        for item in test:
            f_test.write(item + '\n')

    print('数据集划分成功！')
