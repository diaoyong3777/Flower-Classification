# 训练配置
import torch

class TrainingConfig:
    model_name = 'resnet50'# 【越大越好】
    filename = 'resnet50.pt'  # 权重保存位置
    feature_extract = True # 【对应2次训练的思路可以先True，再False】

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #gpu
    train_on_gpu = torch.cuda.is_available()
    batch_size = 128

    # 数据增强参数,大一点会更强
    Resize = 96  # 统一大小
    CenterCrop = 64  # 裁剪大小

    learning_rate = 0.001
    step_size = 2 # 每5步，衰减到原来的0.2
    gamma = 0.1
    num_epochs = 10 # 次数


training_config = TrainingConfig()

if not training_config.train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')