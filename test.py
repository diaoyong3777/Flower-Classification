from base import *
from config import training_config

# 消除libpng warning: iCCP: known incorrect sRGB profile
warnings.filterwarnings("ignore", category=UserWarning, message="libpng")


### 读取标签的实际名字
with open('cat_to_name.json', 'r') as f:  # json文件以字典存放文件夹编号和花卉名称的对应关系【用于最后名称展示】
    cat_to_name = json.load(f)

## 测试

### 加载训练好的模型
model_ft, input_size = initialize_model(training_config.model_name, 102, training_config.feature_extract, use_pretrained=True)

# GPU模式
model_ft = model_ft.to(training_config.device)

# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
print(f"当前模型准确率: {best_acc}")
model_ft.load_state_dict(checkpoint['state_dict'])
# print(f'模型加载成功，最佳准确率: {best_acc:.4f}')

### #数据加载，自定义一个dataloader
### 在test.py中直接定义自定义数据集类
class ImageFolderWithPaths(datasets.ImageFolder):
    """自定义数据集类，返回图片路径"""
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path

### 重新创建验证集的数据加载器（只针对测试）
# 使用自定义数据集类创建验证集
valid_dataset_with_paths = ImageFolderWithPaths(
    os.path.join(data_dir, 'valid'),
    data_transforms['valid']
)

# 创建新的验证集数据加载器
valid_dataloader_with_paths = torch.utils.data.DataLoader(
    valid_dataset_with_paths,
    batch_size=training_config.batch_size,
    shuffle=True  # 测试时不需要打乱
)





# 迭代器，一次获取一个batch的测试数据，next获取下一个【下次运行就是下一批数据了】
dataiter = iter(valid_dataloader_with_paths)
images, labels,paths = dataiter.__next__()  # 获取图像+标签 【Pytorch版本更新：next改为__next__】

# 验证模式，不做更新
model_ft.eval()

# 数据传入cuda还是直接传
if training_config.train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)


### 得到概率最大的类别

_, preds_tensor = torch.max(output, 1)  # 得到概率最大的类别
# 用gpu还得先取到cpu再将tensor转成数组adarray形式【画图matplotlib只支持数组】【torch对应tensor对应gpu，numpy对应cpu】
preds = np.squeeze(preds_tensor.numpy()) if not training_config.train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

mapping = [
    1, 10, 100, 101, 102, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60,
    61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99
] # 类别0~101要对应文件夹序号1~102

preds = torch.tensor(mapping)[preds] # 将preds作为索引得到map(先转成tensor)元素
labels = torch.tensor(mapping)[labels]



### 预测结果展示

# # 图像数据转换成原格式
# def im_convert(tensor):
#     """展示数据"""
#     # 利用torch的dataloader取得，tensor结构，需要转到cpu转换成数组格式
#     image = tensor.to("cpu").clone().detach()
#     image = image.numpy().squeeze()  # squeeze()压缩
#
#     image = image.transpose(1, 2, 0)
#     image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406)) # 反标准化
#     image = image.clip(0, 1)  # 防止数值越界或异常
#     return image
#
#
# fig = plt.figure(figsize=(20, 20))
# columns = 4
# rows = 2
#
# for idx in range(columns * rows):  # 展示4×2个图像
#     ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
#     plt.imshow(im_convert(images[idx]))
#     # 利用json得到数字对应的花卉名称。预测正确绿色，错误为红色 格式：预测(实际)
#     ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
#                  color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
# plt.show()  # 开始展示


# ### 方法2：获取原始图片路径并显示高清原图
# def get_original_images(batch_images, batch_labels, dataset=image_datasets['valid']):
#     """获取原始高清图片"""
#     original_images = []
#     # 获取这个batch中每个样本的原始文件路径
#     for i in range(len(batch_labels)):
#         # 获取原始图片路径
#         img_path = dataset.imgs[batch_labels[i]][0]  # (path, class_index)
#         # 加载原图
#         original_img = Image.open(img_path)
#         original_images.append(original_img)
#     return original_images
#
#
# # 获取原始高清图片
# original_imgs = get_original_images(images, labels.cpu().numpy())
#
# ### 预测结果展示 - 使用原图
# fig = plt.figure(figsize=(20, 20))
# columns = 4
# rows = 2
#
# for idx in range(columns * rows):
#     ax = fig.add_subplot(rows, columns, idx + 1)
#     ax.axis('off')  # 去掉坐标轴
#
#     # 显示原始高清图片
#     ax.imshow(original_imgs[idx])
#
#     # 设置标题
#     pred_label = cat_to_name[str(preds[idx])]
#     true_label = cat_to_name[str(labels[idx].item())]
#     color = "green" if pred_label == true_label else "red"
#
#     ax.set_title(f"{pred_label}\n({true_label})", color=color, fontsize=12, pad=10)
#
# plt.tight_layout()
# plt.show()


### 获取并统一处理原始图片【最终版】
### 获取并统一处理原始图片 - 使用更小的尺寸
### 图片处理函数
def get_unified_original_images_from_paths(image_paths, target_size=(180, 180)):
    unified_images = []
    for img_path in image_paths:
        original_img = Image.open(img_path)
        original_img.thumbnail(target_size, Image.Resampling.LANCZOS)
        unified_img = Image.new('RGB', target_size, (255, 255, 255))
        x = (target_size[0] - original_img.size[0]) // 2
        y = (target_size[1] - original_img.size[1]) // 2
        unified_img.paste(original_img, (x, y))
        unified_images.append(unified_img)
    return unified_images


unified_imgs = get_unified_original_images_from_paths(paths, target_size=(180, 180))

### 显示结果
fig = plt.figure(figsize=(16, 8))
columns = 4
rows = 2

for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1)
    ax.imshow(unified_imgs[idx])
    ax.axis('off')

    pred_flower_id = preds[idx].item()
    true_flower_id = labels[idx].item()
    pred_name = cat_to_name[str(pred_flower_id)]
    true_name = cat_to_name[str(true_flower_id)]

    is_correct = (pred_flower_id == true_flower_id)
    color = "green" if is_correct else "red"
    result = " ✓" if is_correct else " ✗"

    title_text = f"P: {pred_name}{result}\nT: {true_name}"
    ax.set_title(title_text, color=color, fontsize=10, fontweight='bold', pad=6, linespacing=1.2)

plt.suptitle('Flower Classification', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=2.0)
plt.show()

correct_count = sum(1 for i in range(columns * rows) if preds[i] == labels[i])
print(f"\n显示样本统计:")
print(f"正确预测: {correct_count}/{columns * rows}")
print(f"准确率: {correct_count / (columns * rows) * 100:.1f}%")