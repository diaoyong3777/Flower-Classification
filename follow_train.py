from base import *

# # 将所有参数解冻,注意！！！！！！这里会显示只训练输出层，但其实都训练了
# 【或者在配置中设置False即可,为了方便可以直接这样免得改。如果要只训练输出层，把下面注释掉就行】
for param in model_ft.parameters():
    param.requires_grad = True


# 加载训练好的模型
checkpoint = torch.load(filename) # 从文件获取模型
best_acc = checkpoint['best_acc'] # 将之前的最佳准确率作为初始值
print(f"当前模型准确率: {best_acc}")
model_ft.load_state_dict(checkpoint['state_dict']) # 加载模型

# 不断优化
train_model(best_acc=best_acc)