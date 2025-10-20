## 花卉分类

教程：https://www.bilibili.com/video/BV1xf421D7iD/?spm_id_from=333.788.videopod.episodes&vd_source=4a427baba384d9f5d66ea328f421e654&p=6

GitHub项目地址：https://github.com/diaoyong3777/Flower-Classification

参考博客1：[https://2048ai.net/682fe90e606a8318e85a0171.html?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-6-119777597-blog-148049994.235%5Ev43%5Epc_blog_bottom_relevance_base7&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-6-119777597-blog-148049994.235%5Ev43%5Epc_blog_bottom_relevance_base7&utm_relevant_index=13#devmenu2](https://2048ai.net/682fe90e606a8318e85a0171.html?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-6-119777597-blog-148049994.235^v43^pc_blog_bottom_relevance_base7&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-6-119777597-blog-148049994.235^v43^pc_blog_bottom_relevance_base7&utm_relevant_index=13#devmenu2)

参考博客2：https://blog.csdn.net/qiuzitao/article/details/108644082?login=from_csdn





### 1，环境

cuda==12.4

python==3.9

torch==cuda12.4

jupyter notebook==6.1.4

matplotlib



### 2，目录结构-文件使用

flower_data：数据集

cat_to_name.json：存放文件夹序号对应的花卉名称

best.pt：训练好的模型保存位置

训练好的模型：可以把训练好的存放在这自行整理



- jupyter

main.ipynb：项目的核心源代码

easy.ipynb：简化升级版



- pycharm

base.py：存放函数和简单数据处理

config.py：配置，可以修改训练方案

first_train.py：第一次训练，先冻结住除输出层外的部分【配置中的feature_extract = True】。

follow_train.py：后续训练，根据需要自行修改配置然后训练优化

test.py：测试训练好的模型

app.py：可视化界面上传图片进行花卉分类



- pt

resnet18.pt【66%】

resnet50.pt【75%】

resnet152.pt【40%】



【还有部分其余文件都是一些无关的系统自己生成的】

『pt文件——123云盘分享』pt 链接：https://www.123865.com/s/idzcjv-zXNzH











### 3，项目部分说明

![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760790054032-45a5fb93-9ace-478a-bdbd-6bacaf501882.png)

【冻结一部分】

![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760790997176-5c8e99fd-0171-4d6f-8e8d-73db4194ade0.png)

【修改输出大小】

![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760834527186-71a12637-f03a-4e06-93de-ccdefac169b5.png)

【取最佳的参数】

![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760838580057-914f91b0-07f9-4b14-b388-7ea0bb17cedc.png)

【旧版本警告，可忽略】【会下载resnet，不要点ctrl+c中断了】

![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760847846883-a4483b4d-5a00-449e-bdd4-a505adcd11b4.png)

torch是（通道，宽度，高度）。用PIL展示图片需要转成（宽度，高度，通道）



![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760849766602-ec933f4c-c5f8-4f84-ab1b-ec07aa0131a4.png)

一批一批地，中间结果放到内存或者显存，用完就删了，不存到主存中。







### 4，扩展应用

我想投入到应用当中。就是我给他一张图片，他给我返回名称和置信度。设计一个这样的交互界面

![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760867343047-4373258c-b16b-48cf-aa0c-ef09370e6f60.png)

【用FALSK，Gradio有版本问题】





### 5，bug说明

#### 文件夹索引不对应

【发现：老是预测错，进一步观察发现 哪怕预测对了，去valid文件夹也找不到对应的图片】

![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760878530160-64bde17b-e6cd-48fe-b10c-23771ae66c9b.png)

【实际的类别序号是1~102】

【预测值和标签返回的都是0~101】

【我们idx+60实验发现出现报错】

【修改：preds+1、labels+1】





——————————还是错误——————————

真正的错误逻辑：

preds/labels——文件夹——名称

​          1        ——    1    —— 花1

​          10      ——    2    —— 花2

​        100     ——     3    —— 花3

​        101      ——    4    —— 花4



类似于这样：

![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760880986140-d7f00d91-67d0-4387-b7d7-8827f997995c.png)





【dataloader的shuffle=False，不打乱顺序，便于调试】

![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760919564503-0479c33e-8f5a-4fcb-a0a1-8ebfd4fc6004.png)

![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760919521224-55ac887a-cb21-4034-9b2c-19a06d27edd2.png)

![img](https://cdn.nlark.com/yuque/0/2025/png/36186524/1760919539002-da4d0d55-ee5c-4faa-80a4-9bbfdc3daaf4.png)

【第一批正确，我将idx+8 。第二批就错了，第二批的图片可以在10文件夹找到】

修改：

preds = torch.tensor(mapping)[preds] # 将preds作为索引得到map(先转成tensor)元素

lables = torch.tensor(mapping)[labels]

preds,lables

【preds本来是数组被转换为了tensor，所以取指时要加上.item()，即preds[index].item()，preds[index]是tensor(num)】







#### best_acc会毁掉初始模型

```plain
# 不断优化
train_model(best_acc=best_acc)
```

【def train_model(model=model_ft, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer_ft, num_epochs=training_config.num_epochs, is_inception=False, filename=filename,best_acc = 0):】
