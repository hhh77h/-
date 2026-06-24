# 深度学习课程设计：基于 ResNet18 的菜品图像分类系统

## 项目概述
- 任务：五类菜品图像分类（dumplings / fried_rice / kung_pao_chicken / mapo_tofu / sweet_and_sour_pork）
- 最终模型：ResNet18（迁移学习，ImageNet 预训练，分类头替换为 5 类）
- 基准模型：MLP（图像缩放后展平输入的传统基线）

## 目录说明（工程根目录）
- DL课程设计.ipynb：课程设计报告（图表在本地路径中直接引用）
- custom_food_classifier/：自建数据集 + ResNet18 训练/评估核心代码
- scraped_food_data/outputs/：ResNet18 训练输出（best_model.pth / history.json / summary.json 等）
- baseline_model/：MLP baseline 输出（best_model.joblib / 指标 / 混淆矩阵等）
- report_figures/：ResNet18 图表（曲线、混淆矩阵、可视化分析等）
- baseline_report_figures/：MLP baseline 图表
- image.py：ResNet18 图表绘制函数集合（供 ipynb 调用）
- baseline_model.py：MLP baseline 训练脚本（可直接运行）
- baseline_image.py：MLP baseline 图表绘制脚本（可直接运行）
- requirements_scraped_food.txt：依赖列表

## 已有实验结果（当前工程真实输出）
### ResNet18（最终模型）
- 训练集：1159，验证集：247，测试集：254
- 最佳验证准确率：89.07%
- 测试集准确率：92.91%
- 分类报告：scraped_food_data/outputs/classification_report.csv（若你保存了该文件）

### Baseline（MLP）
- 训练集准确率：94.56%
- 验证集准确率：46.56%
- 测试集准确率：42.91%
- 说明：训练集远高于验证/测试，存在明显过拟合；也验证了卷积网络在图像空间特征建模上的优势。

## 运行方式（Windows PowerShell）
### 1) 安装依赖
```powershell
cd "e:\深度学习课设"
pip install -r requirements_scraped_food.txt
```

### 2) 训练 ResNet18（自建数据集）
```powershell
cd "e:\深度学习课设"
python -m custom_food_classifier.train --dataset-root ./scraped_food_data/split --output-root ./scraped_food_data/outputs --epochs 10
```

### 3) 训练 Baseline（MLP）
```powershell
cd "e:\深度学习课设"
python baseline_model.py
```

### 4) 生成图表
- ResNet18 图表（保存到 report_figures/）：
```powershell
cd "e:\深度学习课设"
python -c "from image import generate_all_report_figures; print(generate_all_report_figures())"
```

- Baseline 图表（保存到 baseline_report_figures/）：
```powershell
cd "e:\深度学习课设"
python baseline_image.py
```

