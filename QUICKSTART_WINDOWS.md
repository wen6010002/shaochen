# Windows 快速开始指南

## ✅ 检查清单

在开始训练前，请确认以下项目：

### 1. 文件结构检查
```
你的项目文件夹/
├── pointnet.pytorch-master/       ✓ 必需
│   ├── pointnet/
│   │   ├── dataset.py            ✓ 已修改
│   │   └── model.py              ✓
│   └── utils/
│       └── train_classification.py  ✓ 已修改
├── modelnet40_normal_resampled/   ✓ 必需
│   ├── airplane/                 ✓ 40个类别文件夹
│   ├── trainval.txt              ✓ 已创建
│   └── test.txt                  ✓ 已创建
└── RUN_TRAINING_WINDOWS.bat       ✓ 已创建
```

### 2. Python 环境检查

打开 CMD，依次运行：

```bash
# 检查 Python 版本（需要 3.6+）
python --version

# 检查依赖是否已安装
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import tqdm; print('tqdm: OK')"
python -c "import plyfile; print('plyfile: OK')"
```

如果报错，运行安装命令：
```bash
pip install torch tqdm plyfile numpy
```

## 🚀 三步开始训练

### 步骤 1: 打开命令提示符
- 按 `Win + R`
- 输入 `cmd`
- 回车

### 步骤 2: 进入项目目录
```bash
cd C:\你的路径\项目文件夹
```

### 步骤 3: 运行训练
```bash
# 快速测试（2轮，约5-10分钟 CPU / 1分钟 GPU）
RUN_TRAINING_WINDOWS.bat

# 完整训练（250轮）
RUN_TRAINING_WINDOWS.bat 250
```

## 📊 训练过程

你将看到类似输出：

```
========================================
PointNet Training on ModelNet40
========================================

Configuration:
- Dataset: modelnet40_normal_resampled
- Epochs: 2
- Batch Size: 32
- Points per sample: 2500

Starting training...

Using device: cuda (或 cpu)
9843 2468
classes 40
[0: 0/307] train loss: 3.689120 accuracy: 0.031250
[0: 0/307] test loss: 3.682451 accuracy: 0.062500
...
final accuracy 0.XXXX

========================================
Training completed!
Models saved in: pointnet.pytorch-master\utils\cls\
========================================
```

## 🔧 常见问题快速修复

### Q1: 提示找不到模块
```bash
pip install torch tqdm plyfile numpy
cd pointnet.pytorch-master
pip install -e .
```

### Q2: 找不到数据集
- 确认 `modelnet40_normal_resampled` 文件夹在项目根目录
- 确认里面有 40 个子文件夹（airplane, bathtub, ...）

### Q3: 内存不足
编辑批处理文件，改小参数：
```batch
python train_classification.py ... --batchSize 8 --num_points 1024
```

### Q4: CPU训练太慢
- 正常现象，CPU 比 GPU 慢 10-50 倍
- 建议先用 2-5 个 epoch 测试
- 或者使用云端 GPU（Google Colab, Kaggle 等）

## 📈 预期结果

### 训练 2 轮（测试）
- 时间: CPU 10-30分钟 / GPU 1-3分钟
- 准确率: 20-40%（正常，仅测试代码是否运行）

### 训练 250 轮（完整）
- 时间: CPU 数天 / GPU 15-20小时
- 准确率: 86-88%

## 📁 输出文件

训练完成后，模型保存在：
```
pointnet.pytorch-master\utils\cls\cls_model_0.pth
pointnet.pytorch-master\utils\cls\cls_model_1.pth
...
```

每个 `.pth` 文件是一个训练完成的模型，可用于：
- 继续训练: `--model cls/cls_model_X.pth`
- 推理预测
- 模型评估

## 💡 高级使用

### 使用特征变换
```bash
python train_classification.py ... --feature_transform
```
（准确率可能提高 0.5-1%，但训练更慢）

### 从已有模型继续训练
```bash
python train_classification.py ... --model cls\cls_model_10.pth
```

### 调整学习率
编辑 `train_classification.py` 第 94 行：
```python
optimizer = optim.Adam(classifier.parameters(), lr=0.001, ...)
# 改为
optimizer = optim.Adam(classifier.parameters(), lr=0.0005, ...)
```

## 📞 需要帮助？

1. 查看详细文档: `README_WINDOWS.md`
2. 检查错误信息，通常会提示具体问题
3. 确认所有文件按照检查清单准备完毕

---

**已完成的修改**:
- ✅ 数据集加载支持 .txt 格式
- ✅ Windows 路径兼容
- ✅ CPU/GPU 自动检测
- ✅ 创建训练/测试文件列表
- ✅ 提供批处理运行脚本

**可以直接使用，无需额外修改！**
