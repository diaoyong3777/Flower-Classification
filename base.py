'''基础配置'''
## 导包
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
# pip install torchvision    #如果你的电脑没有安装torchvision模块就得去用这个指令安装
from torchvision import transforms, models, datasets  # transforms——数据增强。models——现成的模型。datasets——目录结构
# https://pytorch.org/docs/stable/torchvision/index.html  #模块的官方网址，上面例子有教你怎么用
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

from config import training_config

## 数据读取
data_dir = './flower_data/'  # 数据所在目录
train_dir = data_dir + '/train'  # 训练集
valid_dir = data_dir + '/valid'  # 验证集

## 数据预处理

data_transforms = {
    'train':
        transforms.Compose([  # 按顺序执行下面的操作
            transforms.Resize([training_config.Resize, training_config.Resize]),  # 统一图片大小【别选太大，慢而且可能算力不够】
            transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选，太大了都倒着了不太合适
            transforms.CenterCrop(training_config.CenterCrop),  # 在96×96的图片随机裁剪出64×64的。（随机裁剪得到的数据更多）
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率去翻转，0.5就是50%翻转，50%不翻转
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B【这个和上面一个不常用】
            transforms.ToTensor(),  # 数据转成tensor的格式
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差（拿人家算好的）三个值对应RGB
        ]),
    'valid':  # 实际应用效果以实际图像为准，不需要增强了，而且够用了，"考卷"多了也没必要
        transforms.Compose([
            transforms.Resize([64, 64]),  # 训练数据裁剪后最终是64的，这里直接resize到64
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 要和训练集保持一致的标准化操作
        ]),
}
### 【一般训练集咋做，验证集就咋做】

## 数据加载


# datasets.ImageFolder 加载器 文件夹名作为文件夹下数据的标签
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# train的数据做train对应的预处理，valid同理。os.path.join(data_dir, x)<=>flower_data/train/
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=training_config.batch_size, shuffle=True) for x in
               ['train', 'valid']}  # 数据加载
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}  # 训练集大小
class_names = image_datasets['train'].classes



## 模型配置
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():  # 遍历每个参数，设置反向传播时不计算梯度，不更新参数
            param.requires_grad = False



# 初始化模型函数
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    # 修改这里：判断模型名称是否包含 "resnet"
    if "resnet" in model_name.lower():
        """ Resnet系列模型
        """

        # model_ft = models.model_name(pretrained=use_pretrained) # models.resnet18 模型。pretrained=use_pretrained 使用预训练好的参数
        # models.model_name【Python 会把 model_name 当成 models 模块的一个固定属性，而不是使用变量 model_name 的值】
        # 改为 getattr(models, model_name)【根据字符串 model_name 的值（比如 "resnet18"），获取 models 模块中对应的属性（比如 models.resnet18】
        model_ft = getattr(models, model_name)(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, training_config.feature_extract)  # 是否冻结参数 【下面的自定义输出层没被冻上】
        num_ftrs = model_ft.fc.in_features  # 查询全连接层（输出层）的输入大小
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),  # 自定义输出层，保持输入大小，输出改为102【默认要反向传播更新参数】
                                    nn.LogSoftmax(dim=1))
        input_size = 64  # 配置高可以增加

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


### 参数更新配置

# 初始化模型
model_ft, input_size = initialize_model(training_config.model_name, 102, training_config.feature_extract, use_pretrained=True)

# GPU计算
model_ft = model_ft.to(training_config.device)

# 最佳模型保存（网络结构+参数）
filename = training_config.filename

# 存储并打印要训练哪些参数
params_to_update = model_ft.parameters()  # params_to_update 列表 存储要更新的参数
print("Params to learn:")
if training_config.feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:  # else对应的逻辑是 不改变 params_to_update 列表 直接存储model_ft.parameters()
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

### 优化器

# 优化器设置，常用optim.Adam
optimizer_ft = optim.Adam(params_to_update, lr=training_config.learning_rate)  # 放入要更新的参数 ，学习率=0.01
# 衰减器
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=training_config.step_size, gamma=training_config.gamma)
# 最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()  # 损失函数


### 训练配置


# 模型、数据、损失、优化器、迭代次数（默认25）、参数保存路径
def train_model(model=model_ft, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer_ft, num_epochs=training_config.num_epochs, is_inception=False, filename=filename,best_acc = 0):
    since = time.time()  # 记录当前时间
    model.to(training_config.device)  # 用GPU或CPU训练
    # 记录每次epoch的准确率和损失
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    # 获取学习率，在优化器字典对象里找
    LRs = [optimizer.param_groups[0]['lr']]
    # 把当前的参数记录下来作为初始值，后续有更好的再更新
    best_model_wts = copy.deepcopy(model.state_dict())

    # 一个epoch-1个epoch来遍历【1个epoch对应多次迭代】——————————
    for epoch in range(num_epochs):
        # 打印当前执行到第几个epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证【一次epoch包括训练和验证阶段】——————————
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            # 初始化损失和正确个数
            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍【iter 迭代】————————————
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(training_config.device)  # 将input传入GPU计算
                labels = labels.to(training_config.device)  # 将labels传入GPU计算

                # 先清零再更新【梯度计算会累加上一次的epoch，不会自动清零】
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)  # 输出
                        loss1 = criterion(outputs, labels)  # 计算损失
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:  # resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)  # 拿到最大的输出作为预测值

                    # 只有训练阶段更新权重【训练和验证都前向传播，只有训练需要再反向传播】
                    if phase == 'train':
                        loss.backward()  # 反向传播
                        optimizer.step()  # 参数更新

                # 累加每次迭代的损失和正确个数
                running_loss += loss.item() * inputs.size(0)  # inputs.size(0) 表示 batch
                running_corrects += torch.sum(preds == labels.data)  # 统计正确个数

            # ————————  一次epoch的所有迭代结束 ————————————

            # 计算这次epoch损失和准确率
            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # 总损失/总数据量
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)  # 总正确个数/总数据量

            # 打印这次epoch的训练信息
            time_elapsed = time.time() - since  # 耗费时间
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))  # 损失、准确率

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:  # 更强——验证准确率更高
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())  # 记录当前（最好）的参数
                state = {
                    'state_dict': model.state_dict(),  # 参数字典 key就是各层的名字，值就是训练好的权重
                    'best_acc': best_acc,  # 准确率
                    'optimizer': optimizer.state_dict(),  # 优化器参数
                }
                torch.save(state, filename)  # 一更新就保存到本地文件，可以早停

            # 保存每次epoch的损失和准确率的历史记录
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        # ———————— 一次epoch即将结束，真正结束前再调整一下学习率 ——————————

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))  # 打印当前学习率
        LRs.append(optimizer.param_groups[0]['lr'])  # 保存当前学习率
        print()
        scheduler.step()  # 执行学习率衰减器（每n次epoch衰减一次）

    # —————————————— 所有epoch结束，训练完成 ————————————————

    time_elapsed = time.time() - since  # 训练耗费总时间
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))  # 最佳准确率

    # 训练完后 用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)  # 加载记录好的最佳参数
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  # 返回 最佳模型、历史损失和正确率、学习率















