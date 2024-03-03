import os
import torch
import datetime
import sklearn.metrics as metrics

from .utils_sr import MultiClassFocalLossWithAlpha


def train(model, loader_train, loader_valid, epochs, optimizer):
    # 建立文件夹保存权重
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    # 定义损失函数
    criterion = MultiClassFocalLossWithAlpha()

    # 保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 参数初始化
    train_loss = 0
    best_model = 0

    loss_list_train = []
    loss_list_valid = []

    accuracy_list_train = []
    accuracy_list_valid = []

    # 模型训练
    model.train()

    # cos 学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2)

    for epoch in range(epochs):

        # 初始化准确率
        train_avg_loss = 0
        valid_avg_loss = 0
        train_accuracy = 0
        valid_accuracy = 0

        # 预测标签和真实标签存储
        train_score_list = []
        train_label_list = []
        valid_score_list = []
        valid_label_list = []

        '''模型训练 '''
        for i, (train_images, train_labels) in enumerate(loader_train):
            # 添加到 CUDA 中
            train_images, train_labels = train_images.cuda(), train_labels.cuda()

            # 梯度清零
            optimizer.zero_grad()

            # 获取输出
            train_predicts = model(train_images)

            # 监督损失
            train_loss = criterion(train_predicts, train_labels.long())

            # 反向传播
            train_loss.backward()
            optimizer.step()

            # 计算准确率和平均损失
            train_predict = train_predicts.detach().max(1)[1]
            train_mid_acc = torch.as_tensor(train_labels == train_predict)
            train_accuracy = train_accuracy + torch.sum(train_mid_acc) / len(train_labels)

            train_avg_loss = train_avg_loss + train_loss / len(loader_train)

            # 存储预测值和真实值
            train_score_list.extend(train_predict.cpu().numpy())
            train_label_list.extend(train_labels.cpu().numpy())

        # 更新学习率
        scheduler.step()

        '''模型测试 '''
        with torch.no_grad():

            for i, (valid_images, valid_labels) in enumerate(loader_valid):
                # 添加到 CUDA 中
                valid_images, valid_labels = valid_images.cuda(), valid_labels.cuda()

                # 获取标签
                valid_predicts = model(valid_images)

                # 混合监督损失
                valid_loss = criterion(valid_predicts, valid_labels.long())

                # 计算准确率和平均损失
                valid_predict = valid_predicts.detach().max(1)[1]
                valid_mid_acc = torch.as_tensor(valid_labels == valid_predict)
                valid_accuracy = valid_accuracy + torch.sum(valid_mid_acc) / len(valid_labels)

                valid_avg_loss = valid_avg_loss + valid_loss / len(loader_valid)

                # 存储预测值和真实值
                valid_score_list.extend(valid_predict.cpu().numpy())
                valid_label_list.extend(valid_labels.cpu().numpy())

            # 记录损失
            loss_list_train.append(train_loss.detach().cpu().item())
            loss_list_valid.append(valid_loss.detach().cpu().item())

            # 记录准确率
            accuracy_list_train.append(train_accuracy.detach().cpu().item() / len(loader_train))
            accuracy_list_valid.append(valid_accuracy.detach().cpu().item() / len(loader_valid))

        # 计算召回率
        train_recall = metrics.recall_score(train_score_list, train_label_list, average="macro")
        valid_recall = metrics.recall_score(valid_score_list, valid_label_list, average="micro")

        # 计算 F1 值
        train_f1_score = metrics.f1_score(train_score_list, train_label_list, average="macro")
        valid_f1_score = metrics.f1_score(valid_score_list, valid_label_list, average="macro")

        # 计算精准率
        train_precision = metrics.precision_score(train_score_list, train_label_list, average="macro")
        valid_precision = metrics.precision_score(valid_score_list, valid_label_list, average="macro")

        # 输出结果
        train_avg_loss = train_avg_loss.detach().cpu().item()
        train_accuracy_avg = train_accuracy.detach().cpu().item() / len(loader_train)
        print('训练: Epoch %d, Accuracy %f, Train Loss: %f' % (epoch, train_accuracy_avg, train_avg_loss))

        valid_avg_loss = valid_avg_loss.detach().cpu().item()
        valid_accuracy_avg = valid_accuracy.detach().cpu().item() / len(loader_valid)
        print('验证: Epoch %d, Accuracy %f, Valid Loss: %f' % (epoch, valid_accuracy_avg, valid_avg_loss))

        # 保存最佳验证准确率模型
        if valid_accuracy_avg >= best_model:
            torch.save(model.state_dict(), "save_weights/best_model.pth")
            best_model = valid_accuracy_avg
            print("当前最佳模型已获取")

        # 记录每个 epoch 对应的 train_loss、lr 以及验证集各指标
        with open(results_file, "a") as f:
            info = f"[epoch: {epoch}]\n" \
                   f"train_loss: {train_avg_loss:.6f}\n" \
                   f"valid_loss: {valid_avg_loss:.6f}\n" \
                   f"train_recall: {train_recall:.4f}\n" \
                   f"valid_recall: {valid_recall:.4f}\n" \
                   f"train_F1_score: {train_f1_score:.4f}\n" \
                   f"valid_F1_score: {valid_f1_score:.4f}\n" \
                   f"train_precision: {train_precision:.4f}\n" \
                   f"valid_precision: {valid_precision:.4f}\n" \
                   f"train_accuracy: {train_accuracy_avg:.6f}\n" \
                   f"valid_accuracy: {valid_accuracy_avg:.6f}\n"
            f.write(info + "\n\n")

    # 返回模型
    loss = {'Loss1': loss_list_train, 'Loss2': loss_list_valid,
            'Accuracy1': accuracy_list_train, 'Accuracy2': accuracy_list_valid}

    return model, loss
