from base import *
from config import training_config
# import torch.optim as optim
import copy


def progressive_unfreezing():
    """渐进式解冻：从输出层开始，逐步解冻更多层"""
    # 重新初始化模型（确保从预训练权重开始）
    model_ft, input_size = initialize_model(training_config.model_name, 102,
                                            feature_extract=True, use_pretrained=True)
    model_ft = model_ft.to(training_config.device)

    # 初始状态：只训练输出层
    for param in model_ft.parameters():
        param.requires_grad = False
    for param in model_ft.fc.parameters():
        param.requires_grad = True

    # 定义解冻计划（针对ResNet50优化）
    # 在 gradual_train.py 中修改解冻计划
    unfreeze_schedule = [
        {'layers': ['fc'], 'epochs': 25, 'lr': 0.001},  # 阶段1：充分训练输出层
        {'layers': ['layer4'], 'epochs': 30, 'lr': 0.0005},  # 阶段2：+layer4
        {'layers': ['layer3'], 'epochs': 25, 'lr': 0.0003},  # 阶段3：+layer3
        {'layers': ['layer2'], 'epochs': 20, 'lr': 0.0002},  # 阶段4：+layer2
        {'layers': ['layer1'], 'epochs': 15, 'lr': 0.0001},  # 阶段5：+layer1
        {'layers': ['bn1', 'conv1'], 'epochs': 10, 'lr': 0.00005},  # 阶段6：全部

        # 新增阶段7：整体微调
        {'layers': ['all'], 'epochs': 20, 'lr': 0.00001},  # 阶段7：整体微调
    ]

    # 全局最佳模型和准确率
    global_best_acc = 0
    global_best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_stage = 0

    for stage_idx, stage in enumerate(unfreeze_schedule):
        print(f"\n{'=' * 60}")
        print(f"📋 阶段 {stage_idx + 1}/{len(unfreeze_schedule)}")
        print(f"🔄 解冻层: {stage['layers']}")
        print(f"⏱️ 训练轮数: {stage['epochs']}, 学习率: {stage['lr']}")
        print(f"{'=' * 60}")

        # 解冻当前阶段的层
        for layer_name in stage['layers']:
            for name, param in model_ft.named_parameters():
                if layer_name in name:
                    param.requires_grad = True
                    print(f"✅ 解冻: {name}")

        # 配置优化器（只更新可训练参数）
        trainable_params = []
        for param in model_ft.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        # 在 gradual_train.py 中尝试不同的优化器
        def get_optimizer(model_params, lr):
            # 方法1: AdamW (推荐)
            return optim.AdamW(model_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))

            # 方法2: SGD with momentum
            # return optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=1e-4)

            # 方法3: Adam
            # return optim.Adam(model_params, lr=lr, weight_decay=1e-4)

        # 使用
        optimizer_ft = get_optimizer(trainable_params, stage['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.8)

        print(f"🔧 可训练参数数量: {len(trainable_params)}")
        print(f"🎯 当前学习率: {stage['lr']}")

        # 训练当前阶段，并获取最佳准确率
        stage_best_acc = global_best_acc  # 传入当前全局最佳作为起点
        model_ft, acc_history, _, _, _, _ = train_model(
            model=model_ft,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer_ft,
            scheduler=scheduler,
            num_epochs=stage['epochs'],
            best_acc=stage_best_acc  # 传入当前最佳，train_model内部会保存最佳模型
        )

        # 检查这个阶段是否产生了新的全局最佳
        if acc_history:
            current_stage_best = max([acc.item() for acc in acc_history])

            # 如果这个阶段产生了新的全局最佳，更新全局最佳模型
            if current_stage_best > global_best_acc:
                global_best_acc = current_stage_best
                global_best_model_wts = copy.deepcopy(model_ft.state_dict())
                best_stage = stage_idx + 1
                print(f"🎉 发现新的全局最佳! 阶段{best_stage}, 准确率: {global_best_acc:.4f}")

                # 立即保存当前最佳模型
                best_state = {
                    'state_dict': global_best_model_wts,
                    'best_acc': global_best_acc,
                    'best_stage': best_stage,
                    'unfreeze_schedule': unfreeze_schedule
                }
                best_filename = f'gradual_best_stage{best_stage}_{global_best_acc:.4f}.pt'
                torch.save(best_state, best_filename)
                print(f"💾 最佳模型已保存: {best_filename}")
            else:
                print(f"📊 阶段最佳: {current_stage_best:.4f}, 全局最佳: {global_best_acc:.4f} (阶段{best_stage})")

    # 训练结束后，加载全局最佳模型权重
    model_ft.load_state_dict(global_best_model_wts)
    print(f"\n🔍 加载全局最佳模型 (阶段{best_stage}, 准确率: {global_best_acc:.4f})")

    # 保存最终的最佳模型
    final_state = {
        'state_dict': global_best_model_wts,
        'best_acc': global_best_acc,
        'best_stage': best_stage,
        'unfreeze_schedule': unfreeze_schedule
    }
    final_filename = f'gradual_unfreeze_final_best_{global_best_acc:.4f}.pt'
    torch.save(final_state, final_filename)

    return model_ft, global_best_acc


# 添加直接执行的功能
if __name__ == "__main__":
    print("🚀 开始渐进式解冻训练...")
    print(f"📊 使用模型: {training_config.model_name}")
    print(f"🎯 目标类别: 102类花卉")
    print(f"🖼️ 图像尺寸: {training_config.Resize}x{training_config.Resize}")
    print(f"📦 批次大小: {training_config.batch_size}")
    print("💡 注意: 将保存验证集精度最高的模型")

    final_model, final_accuracy = progressive_unfreezing()

    print(f"\n🎉 渐进式解冻训练完成！")
    print(f"🏆 最终最佳准确率: {final_accuracy:.4f}")
    print(f"💾 最佳模型文件: gradual_unfreeze_final_best_{final_accuracy:.4f}.pt")
