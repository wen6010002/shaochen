# PointNet 训练指南 - Anaconda 版本

> **完整的 Anaconda/Miniconda 使用指南**

## 📑 目录

1. [环境准备](#环境准备)
2. [安装步骤](#安装步骤)
3. [运行训练](#运行训练)
4. [参数配置](#参数配置)
5. [常见问题](#常见问题)
6. [高级用法](#高级用法)

## 环境准备

### 系统要求

- **操作系统**: Windows 10/11
- **Python**: 3.6-3.10 (推荐 3.8)
- **内存**: 至少 8GB RAM
- **存储**: 至少 10GB 可用空间
- **GPU**: 可选，NVIDIA GPU with CUDA（大幅加速训练）

### 安装 Anaconda

如果还未安装 Anaconda，请选择以下之一：

**选项1: Anaconda（完整版，~500MB）**
- 下载: https://www.anaconda.com/download
- 包含 GUI 工具和常用科学计算包

**选项2: Miniconda（轻量版，~50MB）**
- 下载: https://docs.conda.io/en/latest/miniconda.html
- 仅包含核心功能，按需安装包

**安装注意事项**:
- ✅ 勾选 "Add Anaconda to PATH" （方便使用）
- ✅ 勾选 "Register Anaconda as default Python"
- 安装路径避免中文和空格

## 安装步骤

### 方法1: 使用配置文件（推荐）

#### 步骤 1: 打开 Anaconda Prompt

- **方式A**: 开始菜单 → 搜索 "Anaconda Prompt"
- **方式B**: Win + R → 输入 `cmd` → 回车 → 输入 `conda activate`

#### 步骤 2: 进入项目目录

```bash
cd C:\你的路径\项目文件夹
```

#### 步骤 3: 创建环境

```bash
# 使用提供的配置文件创建环境
conda env create -f environment.yml
```

这将自动：
- 创建名为 `pointnet` 的环境
- 安装 Python 3.8
- 安装 PyTorch、NumPy、tqdm 等依赖
- 安装 plyfile 包

#### 步骤 4: 激活环境

```bash
conda activate pointnet
```

#### 步骤 5: 安装 pointnet 包

```bash
cd pointnet.pytorch-master
pip install -e .
cd ..
```

### 方法2: 手动创建环境

#### 步骤 1: 创建新环境

```bash
conda create -n pointnet python=3.8
```

#### 步骤 2: 激活环境

```bash
conda activate pointnet
```

#### 步骤 3: 安装 PyTorch

**有 NVIDIA GPU（推荐）:**

查看 CUDA 版本：
```bash
nvidia-smi
```

根据 CUDA 版本安装：
```bash
# CUDA 11.8
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

**仅 CPU:**
```bash
conda install pytorch cpuonly -c pytorch
```

#### 步骤 4: 安装其他依赖

```bash
conda install numpy tqdm
pip install plyfile
```

#### 步骤 5: 安装 pointnet 包

```bash
cd pointnet.pytorch-master
pip install -e .
cd ..
```

### 验证安装

运行环境检查脚本：
```bash
CHECK_ENVIRONMENT_ANACONDA.bat
```

或手动检查：
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python test_dataset.py
```

## 运行训练

### 方法1: 批处理脚本（最简单）

在 **Anaconda Prompt** 中：

```bash
# 快速测试（2轮，~5-10分钟）
RUN_TRAINING_ANACONDA.bat

# 标准训练（50轮）
RUN_TRAINING_ANACONDA.bat 50

# 完整训练（250轮）
RUN_TRAINING_ANACONDA.bat 250

# 使用自定义环境名
RUN_TRAINING_ANACONDA.bat 10 my_env_name
```

### 方法2: 手动运行

```bash
# 1. 激活环境
conda activate pointnet

# 2. 进入训练目录
cd pointnet.pytorch-master\utils

# 3. 运行训练
python train_classification.py ^
  --dataset ..\..\modelnet40_normal_resampled ^
  --dataset_type modelnet40 ^
  --nepoch 2 ^
  --batchSize 32 ^
  --num_points 2500
```

### 方法3: Python 脚本

创建 `train.py`:
```python
import os
import sys

# 激活环境（在脚本中）
os.system('conda activate pointnet')

# 设置路径
sys.path.insert(0, 'pointnet.pytorch-master')

# 运行训练
os.chdir('pointnet.pytorch-master/utils')
os.system('python train_classification.py --dataset ../../modelnet40_normal_resampled --dataset_type modelnet40 --nepoch 2')
```

## 参数配置

### 基本参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--dataset` | 必需 | 数据集路径 | `..\..\modelnet40_normal_resampled` |
| `--dataset_type` | shapenet | 数据集类型 | `modelnet40` |
| `--nepoch` | 250 | 训练轮数 | 测试:2, 标准:50, 完整:250 |
| `--batchSize` | 32 | 批次大小 | GPU:32, CPU:8-16 |
| `--num_points` | 2500 | 采样点数 | 2500 |

### 高级参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--workers` | 4 | 数据加载线程数（Windows建议0-2） |
| `--outf` | cls | 模型保存目录 |
| `--model` | '' | 预训练模型路径 |
| `--feature_transform` | False | 使用特征变换（提高精度~1%） |

### 使用示例

**标准训练:**
```bash
python train_classification.py --dataset ..\..\modelnet40_normal_resampled --dataset_type modelnet40 --nepoch 250
```

**使用特征变换:**
```bash
python train_classification.py --dataset ..\..\modelnet40_normal_resampled --dataset_type modelnet40 --feature_transform
```

**从检查点继续:**
```bash
python train_classification.py --dataset ..\..\modelnet40_normal_resampled --dataset_type modelnet40 --model ..\cls\cls_model_50.pth
```

**CPU 优化配置:**
```bash
python train_classification.py --dataset ..\..\modelnet40_normal_resampled --dataset_type modelnet40 --batchSize 8 --num_points 1024 --workers 0
```

**GPU 最大性能:**
```bash
python train_classification.py --dataset ..\..\modelnet40_normal_resampled --dataset_type modelnet40 --batchSize 64 --workers 4 --feature_transform
```

## 输出说明

### 训练过程输出

```
Using device: cuda  # 或 cpu
9843 2468           # 训练集和测试集大小
classes 40          # 类别数

[0: 0/307] train loss: 3.689120 accuracy: 0.031250
[0: 0/307] test loss: 3.682451 accuracy: 0.062500
[0: 10/307] train loss: 3.645123 accuracy: 0.093750
...
```

### 保存的文件

```
pointnet.pytorch-master\utils\cls\
├── cls_model_0.pth      # 第 0 轮训练后的模型
├── cls_model_1.pth      # 第 1 轮训练后的模型
├── cls_model_2.pth
└── ...
```

每个 `.pth` 文件包含完整的模型权重，可用于：
- 继续训练
- 模型推理
- 精度评估

## 常见问题

### Anaconda 相关问题

#### Q1: conda 命令找不到

**症状**: `'conda' 不是内部或外部命令`

**原因**: Anaconda 未添加到 PATH

**解决方案**:
1. **使用 Anaconda Prompt** 而不是普通 CMD
2. 或添加到 PATH:
   - 右键 "此电脑" → "属性" → "高级系统设置"
   - "环境变量" → 系统变量 → 编辑 Path
   - 添加: `C:\ProgramData\Anaconda3` 和 `C:\ProgramData\Anaconda3\Scripts`
3. 或在 CMD 中运行:
   ```bash
   C:\ProgramData\Anaconda3\Scripts\activate.bat
   ```

#### Q2: conda activate 不工作

**症状**: `conda activate` 无效，环境没有切换

**解决方案**:

初始化 conda（一次性操作）:
```bash
conda init cmd.exe
conda init powershell
```

然后重启命令提示符。

#### Q3: 环境创建失败

**症状**: `Solving environment: failed`

**解决方案**:

```bash
# 1. 清理缓存
conda clean --all

# 2. 更新 conda
conda update conda

# 3. 使用更宽松的求解器
conda config --set channel_priority flexible

# 4. 重新创建环境
conda env create -f environment.yml
```

#### Q4: 包冲突

**症状**: `PackagesNotFoundError` 或版本冲突

**解决方案**:

```bash
# 方法1: 指定不同的 channel
conda install package_name -c conda-forge

# 方法2: 使用 pip 安装
pip install package_name

# 方法3: 降低版本要求
# 编辑 environment.yml，删除版本限制
```

### 训练相关问题

#### Q5: 内存不足 (OOM)

**症状**: `RuntimeError: CUDA out of memory` 或程序崩溃

**解决方案**:

```bash
# 减小 batch size
--batchSize 16

# 减少采样点数
--num_points 1024

# 减少数据加载线程
--workers 0

# 示例
python train_classification.py --dataset ..\..\modelnet40_normal_resampled --dataset_type modelnet40 --batchSize 8 --num_points 1024 --workers 0
```

#### Q6: 多进程错误

**症状**: Windows 上多进程加载数据出错

**解决方案**:

添加 `--workers 0`:
```bash
python train_classification.py ... --workers 0
```

#### Q7: CUDA 错误

**症状**: `CUDA error: device-side assert triggered`

**解决方案**:

1. **检查 CUDA 版本**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.version.cuda)"
   ```

2. **重新安装匹配的 PyTorch**:
   ```bash
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

3. **使用 CPU 训练**（临时）:
   ```bash
   # 代码会自动检测，如果无 GPU 则使用 CPU
   ```

#### Q8: 数据加载慢

**解决方案**:

```bash
# 1. 增加数据加载线程（如果内存足够）
--workers 4

# 2. 预加载到内存（需要足够 RAM）
# 修改 dataset.py，添加缓存机制

# 3. 使用 SSD 存储数据集
```

## 高级用法

### 自定义训练配置

创建配置文件 `config.py`:

```python
class TrainConfig:
    dataset_path = '..\..\modelnet40_normal_resampled'
    dataset_type = 'modelnet40'
    batch_size = 32
    num_points = 2500
    epochs = 250
    learning_rate = 0.001
    use_feature_transform = True
    workers = 2
    output_dir = 'cls_custom'
```

修改训练脚本使用配置。

### 分布式训练（多 GPU）

如果有多个 GPU:

```python
# 修改 train_classification.py
import torch.distributed as dist
import torch.multiprocessing as mp

# 使用 DataParallel
classifier = torch.nn.DataParallel(classifier)

# 或使用 DistributedDataParallel（更快）
```

### 使用 TensorBoard

安装 tensorboard:
```bash
conda install tensorboard
```

修改训练脚本添加日志:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# 在训练循环中
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Accuracy/train', accuracy, epoch)
```

查看日志:
```bash
tensorboard --logdir runs
```

### 模型推理

创建推理脚本 `inference.py`:

```python
import torch
from pointnet.model import PointNetCls

# 加载模型
model = PointNetCls(k=40)
model.load_state_dict(torch.load('cls/cls_model_249.pth'))
model.eval()

# 推理
with torch.no_grad():
    pred, _, _ = model(points)
    pred_class = pred.argmax()
```

### 导出模型

导出为 ONNX 格式:

```python
import torch

dummy_input = torch.randn(1, 3, 2500)
torch.onnx.export(classifier, dummy_input, "pointnet.onnx")
```

## 性能优化

### Conda 性能优化

#### 使用国内镜像

```bash
# 清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

# 显示镜像地址
conda config --set show_channel_urls yes

# 查看配置
conda config --show channels
```

#### 使用 mamba（更快的包管理器）

```bash
# 安装 mamba
conda install mamba -c conda-forge

# 使用 mamba 代替 conda
mamba env create -f environment.yml
mamba install pytorch -c pytorch
```

mamba 比 conda 快 5-10 倍。

### 训练性能优化

#### 混合精度训练（AMP）

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data in dataloader:
    optimizer.zero_grad()

    with autocast():
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

速度提升 1.5-2 倍，显存减少 ~30%。

#### 梯度累积（模拟大 batch）

```python
accumulation_steps = 4

for i, data in enumerate(dataloader):
    pred, trans, trans_feat = classifier(points)
    loss = F.nll_loss(pred, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 环境管理

### 导出环境

```bash
# 导出完整环境
conda env export > environment_full.yml

# 导出跨平台环境
conda env export --from-history > environment_minimal.yml

# 导出 pip 格式
pip list --format=freeze > requirements.txt
```

### 克隆环境

```bash
# 克隆现有环境
conda create --name pointnet_backup --clone pointnet

# 从导出文件创建
conda env create -f environment_full.yml
```

### 更新环境

```bash
# 更新单个包
conda update numpy

# 更新所有包
conda update --all

# 从文件更新环境
conda env update -f environment.yml --prune
```

### 删除环境

```bash
# 删除环境
conda env remove -n pointnet

# 清理缓存
conda clean --all
```

## 备份和迁移

### 备份项目

需要备份的内容:
1. 数据集（如果自己准备的）
2. 训练好的模型 (`cls/*.pth`)
3. 环境配置 (`environment.yml`)
4. 修改的代码

### 迁移到另一台电脑

```bash
# 1. 复制项目文件夹

# 2. 创建环境
conda env create -f environment.yml

# 3. 激活环境
conda activate pointnet

# 4. 安装 pointnet 包
cd pointnet.pytorch-master
pip install -e .

# 5. 验证
python test_dataset.py
```

## 最佳实践

1. **使用虚拟环境隔离项目**
   ```bash
   conda create -n project_name python=3.8
   ```

2. **保持环境配置文件更新**
   ```bash
   conda env export > environment.yml
   ```

3. **定期清理缓存**
   ```bash
   conda clean --all
   ```

4. **使用版本控制**
   - Git 管理代码
   - 环境配置文件加入版本控制

5. **记录实验结果**
   - 使用 TensorBoard 或 wandb
   - 保存训练日志

6. **定期备份模型**
   ```bash
   # 复制重要的检查点
   copy cls\cls_model_249.pth backup\
   ```

## 故障排查

### 诊断步骤

1. **检查环境**
   ```bash
   conda info
   conda list
   ```

2. **检查 CUDA**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **检查数据集**
   ```bash
   python test_dataset.py
   ```

4. **运行环境检查**
   ```bash
   CHECK_ENVIRONMENT_ANACONDA.bat
   ```

5. **查看错误日志**
   - 完整的错误堆栈
   - 最后几行错误信息

### 获取帮助

1. 查看文档: `QUICKSTART_ANACONDA.md`
2. 运行诊断脚本
3. 检查 Anaconda 文档: https://docs.conda.io/
4. PyTorch 论坛: https://discuss.pytorch.org/

---

**完整的 Anaconda 使用指南，涵盖所有场景！** 🐍
