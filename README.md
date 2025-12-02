# PointNet 分类训练 - Windows 完整版

> **已完成 Windows 适配！支持 pip 和 Anaconda 两种安装方式！**

这是一个已适配 Windows 系统的 PointNet 点云分类训练项目，使用 ModelNet40 数据集。

## 🎯 选择你的环境

### 我使用 Anaconda / Miniconda

**推荐给**：
- 科学计算和深度学习用户
- 需要独立 Python 环境的用户
- 希望自动管理 CUDA 的用户

**快速开始**：
1. 打开 **Anaconda Prompt**
2. 运行 `conda env create -f environment.yml`
3. 运行 `RUN_TRAINING_ANACONDA.bat`

**详细文档**：查看 [QUICKSTART_ANACONDA.md](QUICKSTART_ANACONDA.md)

---

### 我使用系统 Python / pip

**推荐给**：
- 熟悉 pip 的用户
- 不想安装 Anaconda 的用户
- 系统已有合适 Python 版本的用户

**快速开始**：
1. 打开 **CMD** 或 **PowerShell**
2. 运行 `pip install torch tqdm plyfile numpy`
3. 运行 `RUN_TRAINING_WINDOWS.bat`

**详细文档**：查看 [QUICKSTART_WINDOWS.md](QUICKSTART_WINDOWS.md)

---

## 🚀 超快速开始（3步走）

### 使用 Anaconda

```bash
# 1. 创建环境（Anaconda Prompt）
conda env create -f environment.yml

# 2. 检查环境
CHECK_ENVIRONMENT_ANACONDA.bat

# 3. 开始训练
RUN_TRAINING_ANACONDA.bat
```

### 使用 pip

```bash
# 1. 安装依赖（CMD 或 PowerShell）
pip install torch tqdm plyfile numpy
cd pointnet.pytorch-master
pip install -e .

# 2. 检查环境
CHECK_ENVIRONMENT_WINDOWS.bat

# 3. 开始训练
RUN_TRAINING_WINDOWS.bat
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
├── environment.yml                   # ✅ Conda 环境配置
│
├── RUN_TRAINING_ANACONDA.bat         # ✅ Anaconda 训练脚本
├── RUN_TRAINING_WINDOWS.bat          # ✅ pip 训练脚本
│
├── CHECK_ENVIRONMENT_ANACONDA.bat    # ✅ Anaconda 环境检查
├── CHECK_ENVIRONMENT_WINDOWS.bat     # ✅ pip 环境检查
│
├── test_dataset.py                   # ✅ 数据集测试脚本
│
├── README.md                         # 📖 本文档（导航）
├── QUICKSTART_ANACONDA.md            # 📖 Anaconda 快速开始
├── QUICKSTART_WINDOWS.md             # 📖 pip 快速开始
├── README_ANACONDA.md                # 📖 Anaconda 完整文档
├── README_WINDOWS.md                 # 📖 pip 完整文档
└── CHANGELOG.md                      # 📖 代码修改日志
```

## 📋 功能特性

- ✅ **Windows 完全兼容** - 路径处理自动适配
- ✅ **CPU/GPU 自动检测** - 无需手动配置
- ✅ **支持 .txt 数据格式** - 适配 ModelNet40 数据集
- ✅ **两种安装方式** - pip 或 Anaconda 任选
- ✅ **一键运行脚本** - 批处理文件简化操作
- ✅ **完整中文文档** - 详细说明每个步骤
- ✅ **环境检查工具** - 自动诊断配置问题

## 📖 文档导航

### 按使用环境选择

| 你使用的环境 | 快速开始 | 完整文档 |
|-------------|---------|---------|
| **Anaconda/Miniconda** | [QUICKSTART_ANACONDA.md](QUICKSTART_ANACONDA.md) | [README_ANACONDA.md](README_ANACONDA.md) |
| **系统 Python / pip** | [QUICKSTART_WINDOWS.md](QUICKSTART_WINDOWS.md) | [README_WINDOWS.md](README_WINDOWS.md) |

### 按需求选择

| 文档 | 用途 | 适合人群 |
|------|------|----------|
| **README.md** (本文件) | 项目概述和导航 | 所有用户（先看这个） |
| **QUICKSTART_ANACONDA.md** | Anaconda 三步开始 | Anaconda 新手 |
| **QUICKSTART_WINDOWS.md** | pip 三步开始 | pip 新手 |
| **README_ANACONDA.md** | Anaconda 完整手册 | 需要详细说明（Anaconda） |
| **README_WINDOWS.md** | pip 完整手册 | 需要详细说明（pip） |
| **CHANGELOG.md** | 代码修改详情 | 开发者/审查者 |

## 🎯 训练选项对比

### Anaconda 用户

```bash
# 快速测试（2轮）
RUN_TRAINING_ANACONDA.bat

# 标准训练（50轮）
RUN_TRAINING_ANACONDA.bat 50

# 完整训练（250轮）
RUN_TRAINING_ANACONDA.bat 250
```

### pip 用户

```bash
# 快速测试（2轮）
RUN_TRAINING_WINDOWS.bat

# 标准训练（50轮）
RUN_TRAINING_WINDOWS.bat 50

# 完整训练（250轮）
RUN_TRAINING_WINDOWS.bat 250
```

## 📊 预期结果

| 训练配置 | 训练轮数 | CPU时间 | GPU时间 | 准确率 |
|---------|---------|---------|---------|--------|
| 快速测试 | 2 | 10-30分钟 | 1-3分钟 | 20-40% |
| 标准训练 | 50 | 2-4小时 | 15-30分钟 | 70-80% |
| 完整训练 | 250 | 5-10天 | 15-20小时 | 86-88% |

## 🔧 环境对比

### Anaconda vs pip

| 特性 | Anaconda | pip |
|------|----------|-----|
| 环境隔离 | ✅ 完整隔离 | ⚠️ 需配合 venv |
| 安装速度 | ✅ 二进制包，快 | ⚠️ 需编译，慢 |
| CUDA 配置 | ✅ 自动配置 | ⚠️ 手动选择 |
| 包管理 | ✅ 自动解决冲突 | ⚠️ 可能冲突 |
| 适合人群 | 科学计算、深度学习 | 通用开发 |
| 磁盘占用 | ⚠️ 较大（~2GB） | ✅ 较小 |

**推荐**：
- 新手、深度学习项目 → Anaconda
- 熟悉 Python、系统简洁 → pip

## ❓ 常见问题

<details>
<summary><b>Q1: 我应该选择 Anaconda 还是 pip？</b></summary>

**选择 Anaconda 如果**：
- 你是深度学习新手
- 需要独立的项目环境
- 想要自动管理 CUDA
- 有足够的磁盘空间

**选择 pip 如果**：
- 你熟悉 Python 和 pip
- 系统已有合适的 Python
- 磁盘空间有限
- 不需要多个环境隔离
</details>

<details>
<summary><b>Q2: 两种方式可以共存吗？</b></summary>

可以！但注意：
- Anaconda 有自己的 Python 和包管理
- 在 Anaconda Prompt 中使用 conda
- 在普通 CMD 中使用 pip
- 不要混用两种方式安装同一个包
</details>

<details>
<summary><b>Q3: 如何检查我的环境？</b></summary>

**Anaconda 用户**：
```bash
CHECK_ENVIRONMENT_ANACONDA.bat
```

**pip 用户**：
```bash
CHECK_ENVIRONMENT_WINDOWS.bat
```
</details>

<details>
<summary><b>Q4: 训练太慢怎么办？</b></summary>

1. 确认使用 GPU（如果有）
2. 减少 batch size 和 num_points
3. 先用少量 epoch 测试
4. 考虑使用云端 GPU（Colab、Kaggle）
</details>

<details>
<summary><b>Q5: 找不到数据集？</b></summary>

确认：
1. `modelnet40_normal_resampled` 文件夹在项目根目录
2. 文件夹内有 40 个子文件夹（airplane, bathtub等）
3. 存在 `trainval.txt` 和 `test.txt` 文件
</details>

## 🔍 快速诊断

### 步骤 1: 确认你的环境类型

```bash
# 检查是否安装了 Anaconda
conda --version

# 如果成功显示版本 → 使用 Anaconda 文档
# 如果提示找不到命令 → 使用 pip 文档
```

### 步骤 2: 运行环境检查

**Anaconda**:
```bash
CHECK_ENVIRONMENT_ANACONDA.bat
```

**pip**:
```bash
CHECK_ENVIRONMENT_WINDOWS.bat
```

### 步骤 3: 根据错误信息修复

检查脚本会告诉你缺少什么，按提示安装即可。

## 🎓 学习路径

### 新手用户（推荐顺序）

1. **阅读本文档** (README.md) - 了解项目整体
2. **选择环境** - Anaconda 或 pip
3. **快速开始** - 跟随 QUICKSTART 文档
4. **运行测试** - 2 个 epoch 验证能否运行
5. **查看完整文档** - 需要时参考 README_ANACONDA 或 README_WINDOWS

### 有经验用户

1. 直接查看对应的 QUICKSTART 文档
2. 运行环境检查脚本
3. 开始训练
4. 需要时参考完整文档

### 开发者

1. 查看 CHANGELOG.md 了解修改内容
2. 阅读源代码注释
3. 根据需要自定义配置

## 🛠️ 技术细节

### 已修改的内容

1. **数据加载** (`pointnet/dataset.py:167-202`):
   - ✅ 支持 `.txt` 格式（ModelNet40）
   - ✅ 路径处理 Windows 兼容（`os.path.join`）
   - ✅ 类别名提取修正

2. **训练脚本** (`utils/train_classification.py:94-152`):
   - ✅ CPU/GPU 自动检测
   - ✅ 设备自适应传输（`.to(device)`）
   - ✅ 移除硬编码 CUDA 调用

3. **数据集配置**:
   - ✅ 创建 `trainval.txt` (9843 样本)
   - ✅ 创建 `test.txt` (2468 样本)

4. **新增文件**:
   - ✅ Anaconda 环境配置（environment.yml）
   - ✅ 两套运行脚本（Anaconda + pip）
   - ✅ 两套检查脚本（Anaconda + pip）
   - ✅ 完整中文文档

详见 **CHANGELOG.md** 了解完整修改日志。

## ✅ 就绪检查清单

### Anaconda 用户

- [ ] Anaconda 或 Miniconda 已安装
- [ ] 运行 `conda env create -f environment.yml`
- [ ] `conda activate pointnet` 能成功激活
- [ ] `CHECK_ENVIRONMENT_ANACONDA.bat` 全部通过
- [ ] `modelnet40_normal_resampled` 文件夹存在
- [ ] `trainval.txt` 和 `test.txt` 存在

全部完成？运行 `RUN_TRAINING_ANACONDA.bat`！

### pip 用户

- [ ] Python 3.6+ 已安装
- [ ] 运行 `pip install torch tqdm plyfile numpy`
- [ ] 运行 `cd pointnet.pytorch-master && pip install -e .`
- [ ] `CHECK_ENVIRONMENT_WINDOWS.bat` 全部通过
- [ ] `modelnet40_normal_resampled` 文件夹存在
- [ ] `trainval.txt` 和 `test.txt` 存在

全部完成？运行 `RUN_TRAINING_WINDOWS.bat`！

## 📞 获取帮助

### 文档索引

- 🐍 Anaconda 问题 → [README_ANACONDA.md](README_ANACONDA.md)
- 🐍 Anaconda 快速开始 → [QUICKSTART_ANACONDA.md](QUICKSTART_ANACONDA.md)
- 📦 pip 问题 → [README_WINDOWS.md](README_WINDOWS.md)
- 📦 pip 快速开始 → [QUICKSTART_WINDOWS.md](QUICKSTART_WINDOWS.md)
- 🔧 代码修改 → [CHANGELOG.md](CHANGELOG.md)

### 诊断步骤

1. 运行对应的环境检查脚本
2. 查看错误信息
3. 在对应文档的"常见问题"部分查找解决方案
4. 检查文件和目录结构是否正确

### 外部资源

- Anaconda 文档: https://docs.conda.io/
- PyTorch 安装: https://pytorch.org/get-started/locally/
- PointNet 论文: https://arxiv.org/abs/1612.00593
- 原项目: https://github.com/fxia22/pointnet.pytorch

## 📄 许可证

本项目基于原始 PointNet PyTorch 实现进行修改。

原项目: https://github.com/fxia22/pointnet.pytorch

## 🎉 主要特性总结

- ✅ **完全 Windows 兼容** - 所有脚本和代码
- ✅ **双环境支持** - Anaconda 和 pip 都可以
- ✅ **自动化脚本** - 一键运行和检查
- ✅ **详细文档** - 中文说明，涵盖所有场景
- ✅ **已测试验证** - 数据加载和训练流程
- ✅ **开箱即用** - 无需额外修改

---

**已完成 Windows 和 Anaconda 适配，立即开始训练！** 🚀

**不确定用哪个？**
- 新手 → 使用 Anaconda → 看 [QUICKSTART_ANACONDA.md](QUICKSTART_ANACONDA.md)
- 老手 → 使用 pip → 看 [QUICKSTART_WINDOWS.md](QUICKSTART_WINDOWS.md)
