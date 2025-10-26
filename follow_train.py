# from base import *
#
# # # 将所有参数解冻, 注意！！！！！！这里会显示只训练输出层，但其实都训练了
# # 【或者在配置中设置False即可,为了方便可以直接这样免得改。如果要只训练输出层，把下面注释掉就行】
# for param in model_ft.parameters():
#     param.requires_grad = True
#
#
# # 加载训练好的模型
# checkpoint = torch.load(filename) # 从文件获取模型
# best_acc = checkpoint['best_acc'] # 将之前的最佳准确率作为初始值
# print(f"当前模型准确率: {best_acc}")
# model_ft.load_state_dict(checkpoint['state_dict']) # 加载模型
#
# # 不断优化
# train_model(best_acc=best_acc)


from base import *
from config import training_config
import copy

# 重新初始化模型，确保与渐进式训练时结构一致
model_ft, input_size = initialize_model(training_config.model_name, 102,
                                      feature_extract=False, use_pretrained=False)

# 加载渐进式训练的最佳模型
checkpoint = torch.load('pt/98%.pt')  # 替换为你的实际文件名
best_acc = checkpoint['best_acc']
best_stage = checkpoint.get('best_stage', 'unknown')
print(f"📊 加载渐进式训练模型")
print(f"🎯 最佳准确率: {best_acc:.4f}")
print(f"📈 最佳阶段: {best_stage}")

# 加载模型权重
model_ft.load_state_dict(checkpoint['state_dict'])
model_ft = model_ft.to(training_config.device)

# ===== 选择训练策略 =====
strategy = "unfreeze_all"  # 可选: "unfreeze_all", "freeze_output", "custom_freeze"

if strategy == "unfreeze_all":
    # 策略1: 解冻所有层继续训练
    print("🔓 解冻所有层继续训练")
    for param in model_ft.parameters():
        param.requires_grad = True

elif strategy == "freeze_output":
    # 策略2: 冻结输出层，解冻其他层（如果怀疑输出层过拟合）
    print("🔓 解冻除输出层外的所有层")
    for name, param in model_ft.named_parameters():
        if 'fc' in name:  # 冻结输出层
            param.requires_grad = False
            print(f"❄️ 冻结层: {name}")
        else:  # 解冻其他所有层
            param.requires_grad = True

elif strategy == "custom_freeze":
    # 策略3: 自定义冻结策略 - 只解冻深层网络
    print("🔓 自定义冻结: 只解冻layer3和layer4")
    layers_to_unfreeze = ['layer3', 'layer4', 'fc']
    for name, param in model_ft.named_parameters():
        if any(layer in name for layer in layers_to_unfreeze):
            param.requires_grad = True
            print(f"✅ 解冻: {name}")
        else:
            param.requires_grad = False
            print(f"❄️ 冻结: {name}")

# 验证冻结效果
print("\n📋 可训练参数统计:")
trainable_count = 0
total_count = 0
for name, param in model_ft.named_parameters():
    total_count += 1
    if param.requires_grad:
        trainable_count += 1
        print(f"  ✅ {name}")

print(f"\n📊 总共参数层数: {total_count}")
print(f"🎯 可训练参数层数: {trainable_count}")
print(f"📈 可训练比例: {trainable_count/total_count*100:.1f}%")

# 重新配置优化器，只更新可训练参数
params_to_update = []
for param in model_ft.parameters():
    if param.requires_grad:
        params_to_update.append(param)

# 使用更小的学习率进行微调
fine_tune_lr = training_config.learning_rate * 0.1  # 原始学习率的1/10
optimizer_ft = optim.Adam(params_to_update, lr=fine_tune_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=training_config.step_size,
                                    gamma=training_config.gamma)

print(f"🔧 优化器已重新配置")
print(f"📚 学习率: {fine_tune_lr}")

# 继续训练
print("\n🚀 开始继续训练...")
train_model(model=model_ft,
           dataloaders=dataloaders,
           criterion=criterion,
           optimizer=optimizer_ft,
           scheduler=scheduler,
           best_acc=best_acc)

print("🎉 继续训练完成！")
