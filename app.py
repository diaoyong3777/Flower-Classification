from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import json
import io
import base64
from base import *
from config import training_config

app = Flask(__name__)


class FlowerPredictor:
    def __init__(self, model_path='pt/resnet50.pt'):  # 改为你的模型文件名
        with open('cat_to_name.json', 'r') as f:
            self.cat_to_name = json.load(f)

        self.model_ft, _ = initialize_model(training_config.model_name, 102, training_config.feature_extract,
                                            use_pretrained=False)
        self.model_ft = self.model_ft.to(training_config.device)

        checkpoint = torch.load(model_path, map_location=training_config.device)
        best_acc = checkpoint['best_acc']
        print(f"当前模型准确率: {best_acc}")
        self.model_ft.load_state_dict(checkpoint['state_dict'])
        self.model_ft.eval()

        self.transform = transforms.Compose([
            transforms.Resize([training_config.Resize, training_config.Resize]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image, top_k=3):
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(training_config.device)

        with torch.no_grad():
            output = self.model_ft(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top_probs, top_indices = torch.topk(probabilities, top_k)

        # 完整调试信息
        print("=== Flask预测完整调试 ===")
        print(f"1. 原始output形状: {output.shape}")
        print(f"2. 原始output前10个值: {output[0][:10].tolist()}")
        print(f"3. probabilities形状: {probabilities.shape}")
        print(f"4. probabilities前10个值: {probabilities[:10].tolist()}")
        print(f"5. top_indices: {top_indices.tolist()}")
        print(f"6. top_probs: {top_probs.tolist()}")

        # 打印所有102个类别的概率（找到最大值）
        max_prob, max_idx = torch.max(probabilities, 0)
        print(f"7. 最大概率索引: {max_idx.item()}, 概率: {max_prob.item()}")

        # 检查前20个最高概率的索引
        all_top_probs, all_top_indices = torch.topk(probabilities, 20)
        print(f"8. 前20个预测索引: {all_top_indices.tolist()}")
        print(f"9. 前20个预测概率: {all_top_probs.tolist()}")

        results = []
        mapping = [
            1, 10, 100, 101, 102, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42,
            43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60,
            61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
            8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99
        ]  # 类别0~101要对应文件夹序号1~102
        for i in range(top_k):
            raw_index = top_indices[i].item()
            class_idx = str(mapping[raw_index])
            class_name = self.cat_to_name.get(class_idx, f"未知{class_idx}")
            confidence = top_probs[i].item() * 100

            print(
                f"10. 结果{i + 1}详情: raw_index={raw_index}, class_idx='{class_idx}', name='{class_name}', confidence={confidence:.2f}%")

            results.append({
                'name': class_name,
                'confidence': f"{confidence:.2f}%",
                'score': confidence,
                'rank': i + 1,
                'raw_index': raw_index,
                'class_idx': class_idx
            })

        return results


predictor = FlowerPredictor()


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>智能花卉识别系统</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #4CAF50, #45a049);
                color: white;
                padding: 40px 30px;
                text-align: center;
            }

            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 300;
            }

            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }

            .content {
                padding: 40px;
            }

            .upload-section {
                text-align: center;
                margin-bottom: 40px;
            }

            .upload-box {
                border: 3px dashed #4CAF50;
                border-radius: 15px;
                padding: 50px 30px;
                background: #f8fff8;
                transition: all 0.3s ease;
                cursor: pointer;
                margin-bottom: 20px;
            }

            .upload-box:hover {
                border-color: #45a049;
                background: #f0fff0;
            }

            .upload-box i {
                font-size: 3em;
                color: #4CAF50;
                margin-bottom: 15px;
            }

            .upload-box h3 {
                color: #333;
                margin-bottom: 10px;
            }

            .upload-box p {
                color: #666;
                margin-bottom: 20px;
            }

            .file-input {
                display: none;
            }

            .upload-btn {
                background: linear-gradient(135deg, #4CAF50, #45a049);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 50px;
                font-size: 1.1em;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
            }

            .upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
            }

            .upload-btn:active {
                transform: translateY(0);
            }

            .preview-section {
                text-align: center;
                margin: 30px 0;
            }

            #imagePreview {
                max-width: 300px;
                max-height: 300px;
                border-radius: 10px;
                box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                display: none;
            }

            .result-section {
                display: none;
                background: #f8f9fa;
                border-radius: 15px;
                padding: 30px;
                margin-top: 30px;
            }

            .result-section h3 {
                color: #333;
                margin-bottom: 20px;
                text-align: center;
                font-size: 1.5em;
            }

            .result-item {
                background: white;
                padding: 20px;
                margin: 15px 0;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                border-left: 5px solid #4CAF50;
                transition: transform 0.3s ease;
            }

            .result-item:hover {
                transform: translateX(5px);
            }

            .result-item.first {
                border-left-color: #FF6B6B;
                background: linear-gradient(135deg, #fff, #fff5f5);
            }

            .result-item.second {
                border-left-color: #4ECDC4;
            }

            .result-item.third {
                border-left-color: #45B7D1;
            }

            .confidence-bar {
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
                margin: 10px 0;
                overflow: hidden;
            }

            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, #4CAF50, #45a049);
                border-radius: 4px;
                transition: width 1s ease;
            }

            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }

            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #4CAF50;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            @media (max-width: 768px) {
                .container {
                    margin: 10px;
                }

                .content {
                    padding: 20px;
                }

                .header h1 {
                    font-size: 2em;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌺 智能花卉识别系统</h1>
                <p>上传花卉图片，AI智能识别花卉种类</p>
            </div>

            <div class="content">
                <div class="upload-section">
                    <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                        <i>📷</i>
                        <h3>点击选择或拖拽图片</h3>
                        <p>支持 JPG、PNG 格式的图片文件</p>
                        <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="previewImage()">
                    </div>
                    <button class="upload-btn" onclick="predict()">开始识别</button>
                </div>

                <div class="preview-section">
                    <img id="imagePreview" alt="图片预览">
                </div>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>AI正在识别中，请稍候...</p>
                </div>

                <div class="result-section" id="resultSection">
                    <h3>🎯 识别结果</h3>
                    <div id="resultContent"></div>
                </div>
            </div>
        </div>

        <script>
            let currentFile = null;

            function previewImage() {
                const fileInput = document.getElementById('fileInput');
                const preview = document.getElementById('imagePreview');
                const resultSection = document.getElementById('resultSection');

                if (fileInput.files && fileInput.files[0]) {
                    currentFile = fileInput.files[0];
                    const reader = new FileReader();

                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                        resultSection.style.display = 'none';
                    }

                    reader.readAsDataURL(fileInput.files[0]);
                }
            }

            async function predict() {
                if (!currentFile) {
                    alert('请先选择一张图片');
                    return;
                }

                const loading = document.getElementById('loading');
                const resultSection = document.getElementById('resultSection');
                const resultContent = document.getElementById('resultContent');

                loading.style.display = 'block';
                resultSection.style.display = 'none';

                const formData = new FormData();
                formData.append('image', currentFile);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('识别失败');
                    }

                    const results = await response.json();

                    let html = '';
                    results.forEach((item, index) => {
                        const rankClass = index === 0 ? 'first' : index === 1 ? 'second' : 'third';
                        html += `
                            <div class="result-item ${rankClass}">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <h4 style="margin: 0; color: #333;">${item.rank}. ${item.name}</h4>
                                        <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">置信度: ${item.confidence}</p>
                                    </div>
                                    <div style="font-size: 1.5em; color: #4CAF50;">
                                        ${index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'}
                                    </div>
                                </div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${item.score}%"></div>
                                </div>
                            </div>
                        `;
                    });

                    resultContent.innerHTML = html;
                    resultSection.style.display = 'block';

                } catch (error) {
                    alert('识别失败: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                }
            }

            // 拖拽功能
            const uploadBox = document.querySelector('.upload-box');
            uploadBox.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadBox.style.borderColor = '#45a049';
                uploadBox.style.background = '#f0fff0';
            });

            uploadBox.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadBox.style.borderColor = '#4CAF50';
                uploadBox.style.background = '#f8fff8';
            });

            uploadBox.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadBox.style.borderColor = '#4CAF50';
                uploadBox.style.background = '#f8fff8';

                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    document.getElementById('fileInput').files = files;
                    previewImage();
                }
            });
        </script>
    </body>
    </html>
    '''


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        results = predictor.predict(image, top_k=3)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
