# 🌸 花卉分类智能识别系统

一个基于深度学习的花卉图像分类项目，使用PyTorch和预训练的ResNet模型，能够准确识别102种不同花卉品种。

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.4-green)
![License](https://img.shields.io/badge/License-MIT-orange)

## ✨ 项目特色

- 🎯 **高精度识别** - 采用先进的ResNet架构，准确率高达98%
- 🚀 **高效训练** - 支持迁移学习和渐进式训练策略
- 🎨 **友好界面** - 提供Web可视化界面，轻松上传识别
- 📊 **灵活配置** - 模块化设计，支持多种训练方案
- 🔧 **易于部署** - 完整的项目结构和详细文档

## 📁 项目结构

```
Flower-Classification/
├── 🌸 flower_data/                 # 花卉数据集
├── 📊 cat_to_name.json            # 类别标签映射文件
├── ⚙️ config.py                   # 训练配置文件
├── 🔧 base.py                     # 基础函数和数据处理
├── 🚀 first_train.py              # 首次训练（冻结特征层）
├── 🔄 follow_train.py             # 后续训练（全网络调优）
├── 📈 gradual_train.py            # 渐进式训练（推荐）
├── 🧪 test.py                     # 模型测试脚本
├── 🌐 app.py                      # Web交互界面
├── 📓 main.ipynb                  # 完整项目流程
├── 💡 easy.ipynb                  # 简化升级版
└── 📁 pt/                         # 预训练模型
    ├── resnet18.pt     (66%准确率)
    ├── resnet50.pt     (75%准确率)
    ├── resnet50_99%.pt     (99%准确率，采用渐进式训练策略)
    └── resnet152.pt    (40%准确率)
```

## 🛠️ 环境要求

```bash
# 核心依赖
cuda==12.4
python==3.9
torch==cuda12.4

# 可选工具
jupyter notebook==6.1.4
matplotlib
flask
```

## 🚀 快速开始

### 1. 环境配置
```bash
# 克隆项目
git clone https://github.com/diaoyong3777/Flower-Classification.git
cd Flower-Classification


### 2. 数据准备
- 下载花卉数据集到 `flower_data/` 目录
- 确保包含 `train/` 和 `valid/` 子文件夹

### 3. 模型训练

#### 方案一：渐进式训练（推荐👍）
```bash
python gradual_train.py
```
**特点**：逐层解冻训练，达到98%+准确率

#### 方案二：分阶段训练
```bash
# 第一阶段：冻结特征层
python first_train.py

# 第二阶段：全网络调优  
python follow_train.py
```

#### 方案三：Jupyter体验
```python
# 打开 main.ipynb 或 easy.ipynb
# 逐步运行代码块，可视化训练过程
```

### 4. 模型测试
```bash
python test.py
```

### 5. 启动Web应用
```bash
python app.py
```
访问 `http://localhost:5000` 体验花卉识别功能

## 📊 训练策略

### 🎯 迁移学习技术
```python
# 冻结底层特征提取层
for param in model.parameters():
    param.requires_grad = False
    
# 只训练最后的全连接层
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

### 🔄 渐进式解冻
1. **阶段1**：仅训练分类器
2. **阶段2**：解冻最后2个残差块
3. **阶段3**：解冻全部网络层

## 🎨 可视化界面

**功能特点**：
- 📷 支持图片上传
- 🔍 实时识别预测
- 📈 显示置信度
- 🎯 Top-5预测结果


## 🐛 问题解决

### 常见问题
 **类别映射错误** ✅ 已修复
   - 问题：预测标签与文件夹序号不匹配
   - 解决：添加映射校正逻辑



## 📚 参考资源

- 🎥 [视频教程](https://www.bilibili.com/video/BV1xf421D7iD/)
- 📖 [技术博客1](https://2048ai.net/682fe90e606a8318e85a0171.html)
- 📖 [技术博客2](https://blog.csdn.net/qiuzitao/article/details/108644082)

## 🤝 贡献指南

欢迎提交Issue和Pull Request！  
如有问题，请通过以下方式联系：

- 📧 Email: 2151156420@qq.com
- 💬 Issues: [GitHub Issues](https://github.com/diaoyong3777/Flower-Classification/issues)



---

*最后更新: 2024年12月*
