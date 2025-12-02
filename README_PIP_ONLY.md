# PointNet 分类训练 - Windows 版本

> **已完成 Windows 适配！可直接运行！**

这是一个已适配 Windows 系统的 PointNet 点云分类训练项目，使用 ModelNet40 数据集。

## 🚀 快速开始（三步走）

### 1️⃣ 检查环境

双击运行：
```
CHECK_ENVIRONMENT_WINDOWS.bat
```

确保所有检查项通过。如有错误，按提示安装依赖。

### 2️⃣ 开始训练

双击运行：
```
RUN_TRAINING_WINDOWS.bat
```

这将启动快速测试训练（2轮，约5-10分钟）。

### 3️⃣ 查看结果

训练完成后，模型保存在：
```
pointnet.pytorch-master\utils\cls\cls_model_0.pth
pointnet.pytorch-master\utils\cls\cls_model_1.pth
```

## 📁 项目结构

```
项目根目录/
├── pointnet.pytorch-master/          # PointNet 项目
│   ├── pointnet/
│   │   ├── dataset.py               # ✏️ 已修改（支持.txt格式）
│   │   └── model.py
│   └── utils/
│       └── train_classification.py  # ✏️ 已修改（CPU/GPU自适应）
│
├── modelnet40_normal_resampled/      # ModelNet40 数据集
│   ├── airplane/                     # 40个类别文件夹
│   ├── bathtub/
│   ├── ... (共40个类别)
│   ├── trainval.txt                  # ✅ 已创建
│   └── test.txt                      # ✅ 已创建
│
├── CHECK_ENVIRONMENT_WINDOWS.bat     # ✅ 环境检查脚本
├── RUN_TRAINING_WINDOWS.bat          # ✅ 训练启动脚本
├── test_dataset.py                   # ✅ 数据集测试脚本
│
├── README.md                         # 📖 本文档
├── QUICKSTART_WINDOWS.md             # 📖 快速开始指南
├── README_WINDOWS.md                 # 📖 详细使用文档
└── CHANGELOG.md                      # 📖 修改日志
```

## 📋 功能特性

- ✅ **Windows 完全兼容** - 路径处理自动适配
- ✅ **CPU/GPU 自动检测** - 无需手动配置
- ✅ **支持 .txt 数据格式** - 适配 ModelNet40 数据集
- ✅ **一键运行脚本** - 批处理文件简化操作
- ✅ **完整文档** - 中文文档，详细说明
- ✅ **环境检查工具** - 自动诊断配置问题

## 🔧 安装依赖

### 方法1：自动安装（推荐）

```bash
cd pointnet.pytorch-master
pip install torch tqdm plyfile numpy
pip install -e .
```

### 方法2：手动安装

```bash
pip install torch
pip install tqdm
pip install plyfile
pip install numpy
```

## 📖 文档导航

| 文档 | 用途 | 适合人群 |
|------|------|----------|
| **README.md** (本文件) | 项目概述和快速导航 | 所有用户 |
| **QUICKSTART_WINDOWS.md** | 三步快速开始 | 新手用户 |
| **README_WINDOWS.md** | 完整使用手册 | 需要详细说明的用户 |
| **CHANGELOG.md** | 代码修改详情 | 开发者/审查者 |

## 🎯 训练选项

### 快速测试（2轮）
```bash
RUN_TRAINING_WINDOWS.bat
```
- 时间: 5-10分钟 (CPU) / 1分钟 (GPU)
- 用途: 验证代码是否正常运行

### 标准训练（50轮）
```bash
RUN_TRAINING_WINDOWS.bat 50
```
- 时间: 2-4小时 (CPU) / 15-30分钟 (GPU)
- 用途: 初步训练，快速看到效果

### 完整训练（250轮）
```bash
RUN_TRAINING_WINDOWS.bat 250
```
- 时间: 5-10天 (CPU) / 15-20小时 (GPU)
- 用途: 达到论文水平精度 (~86-88%)

### 手动运行（高级）
```bash
cd pointnet.pytorch-master\utils
python train_classification.py ^
  --dataset ..\..\modelnet40_normal_resampled ^
  --dataset_type modelnet40 ^
  --nepoch 250 ^
  --batchSize 32 ^
  --num_points 2500 ^
  --feature_transform
```

## 📊 预期结果

| 训练配置 | 训练轮数 | CPU时间 | GPU时间 | 准确率 |
|---------|---------|---------|---------|--------|
| 快速测试 | 2 | 10-30分钟 | 1-3分钟 | 20-40% |
| 标准训练 | 50 | 2-4小时 | 15-30分钟 | 70-80% |
| 完整训练 | 250 | 5-10天 | 15-20小时 | 86-88% |

## ❓ 常见问题

<details>
<summary><b>Q1: 提示找不到模块</b></summary>

运行安装命令：
```bash
pip install torch tqdm plyfile numpy
cd pointnet.pytorch-master
pip install -e .
```
</details>

<details>
<summary><b>Q2: 找不到数据集</b></summary>

确认：
- `modelnet40_normal_resampled` 文件夹在项目根目录
- 文件夹内有 40 个子文件夹
- 存在 `trainval.txt` 和 `test.txt` 文件
</details>

<details>
<summary><b>Q3: CPU 训练太慢</b></summary>

解决方案：
1. 先用 2-5 个 epoch 测试
2. 使用云端 GPU（Google Colab、Kaggle）
3. 减小 batch size 和 points：`--batchSize 8 --num_points 1024`
</details>

<details>
<summary><b>Q4: 多进程加载错误</b></summary>

在运行命令中添加：`--workers 0`
</details>

<details>
<summary><b>Q5: 内存不足</b></summary>

减小参数：`--batchSize 8 --num_points 1024`
</details>

## 🔍 验证测试

运行数据集测试：
```bash
python test_dataset.py
```

预期输出：
```
✓ Training dataset loaded successfully!
✓ Test dataset loaded successfully!
✓ All tests passed!
```

## 📈 性能对比

### CPU (Intel i7)
- ✓ 可以运行
- ✓ 适合测试
- ✗ 训练太慢（数天）

### GPU (NVIDIA RTX 3080)
- ✓ 快速训练
- ✓ 适合正式训练
- ✓ 15-20小时完成

## 🛠️ 技术细节

### 已修改的内容

1. **数据加载** (`pointnet/dataset.py`):
   - 支持 `.txt` 格式（ModelNet40）
   - 路径处理 Windows 兼容
   - 类别名提取修正

2. **训练脚本** (`utils/train_classification.py`):
   - CPU/GPU 自动检测
   - 设备自适应传输
   - 移除硬编码 CUDA 调用

3. **数据集配置**:
   - 创建 `trainval.txt` (9843 样本)
   - 创建 `test.txt` (2468 样本)

详见 **CHANGELOG.md** 了解完整修改日志。

## 📞 获取帮助

1. 先运行 `CHECK_ENVIRONMENT_WINDOWS.bat` 检查环境
2. 查看对应文档：
   - 快速开始 → `QUICKSTART_WINDOWS.md`
   - 详细说明 → `README_WINDOWS.md`
   - 修改详情 → `CHANGELOG.md`
3. 检查错误信息，通常会提示具体问题

## ✅ 就绪检查清单

在开始训练前，确认：

- [ ] Python 3.6+ 已安装
- [ ] PyTorch 已安装 (`pip install torch`)
- [ ] 其他依赖已安装 (`pip install tqdm plyfile numpy`)
- [ ] pointnet 包已安装 (`cd pointnet.pytorch-master && pip install -e .`)
- [ ] `modelnet40_normal_resampled` 文件夹存在
- [ ] `trainval.txt` 和 `test.txt` 文件存在
- [ ] 运行 `CHECK_ENVIRONMENT_WINDOWS.bat` 全部通过

全部勾选？太好了！运行 `RUN_TRAINING_WINDOWS.bat` 开始训练！

## 📄 许可证

本项目基于原始 PointNet PyTorch 实现进行修改。

原项目: https://github.com/fxia22/pointnet.pytorch

---

**已完成 Windows 适配，可直接使用！**

如有问题，请查看 `README_WINDOWS.md` 或 `QUICKSTART_WINDOWS.md`
