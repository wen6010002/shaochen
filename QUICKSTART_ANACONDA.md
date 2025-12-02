# Anaconda 快速开始指南

> **适用于使用 Anaconda/Miniconda 的用户**

## 🎯 适用人群

- 已安装 Anaconda 或 Miniconda
- 习惯使用 conda 管理 Python 环境
- 需要隔离的 Python 环境

## ✅ 前置要求

- Windows 系统
- Anaconda 或 Miniconda 已安装
- 已解压项目文件和数据集

## 🚀 三步开始（Anaconda 版本）

### 第一步：创建 Conda 环境

打开 **Anaconda Prompt**（不是普通的CMD！），执行：

```bash
# 进入项目目录
cd C:\你的路径\项目文件夹

# 使用配置文件创建环境（推荐）
conda env create -f environment.yml
```

这将创建名为 `pointnet` 的环境，包含所有依赖。

**或者手动创建环境：**

```bash
# 创建新环境
conda create -n pointnet python=3.8

# 激活环境
conda activate pointnet

# 安装依赖
conda install pytorch numpy tqdm -c pytorch
pip install plyfile

# 安装 pointnet 包
cd pointnet.pytorch-master
pip install -e .
cd ..
```

### 第二步：检查环境

在 **Anaconda Prompt** 中运行：

```bash
CHECK_ENVIRONMENT_ANACONDA.bat
```

确保所有检查项通过。

### 第三步：开始训练

**方法1：使用批处理脚本（最简单）**

在 **Anaconda Prompt** 中：
```bash
RUN_TRAINING_ANACONDA.bat
```

**方法2：手动运行**

```bash
# 激活环境
conda activate pointnet

# 进入训练目录
cd pointnet.pytorch-master\utils

# 运行训练
python train_classification.py --dataset ..\..\modelnet40_normal_resampled --dataset_type modelnet40 --nepoch 2
```

## 📋 Conda 环境说明

### environment.yml 内容

```yaml
name: pointnet
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pytorch>=1.7.0
  - numpy
  - tqdm
  - pip
  - pip:
    - plyfile
```

### 常用 Conda 命令

```bash
# 查看所有环境
conda env list

# 激活环境
conda activate pointnet

# 退出环境
conda deactivate

# 删除环境
conda env remove -n pointnet

# 更新环境
conda env update -f environment.yml

# 导出环境
conda env export > my_environment.yml
```

## 🔧 训练选项

### 快速测试（2轮）
```bash
RUN_TRAINING_ANACONDA.bat
```

### 标准训练（50轮）
```bash
RUN_TRAINING_ANACONDA.bat 50
```

### 完整训练（250轮）
```bash
RUN_TRAINING_ANACONDA.bat 250
```

### 使用自定义环境名
```bash
RUN_TRAINING_ANACONDA.bat 10 my_pointnet_env
```

## ❓ 常见问题（Anaconda 版）

### Q1: 找不到 conda 命令

**原因**: Anaconda 未添加到系统 PATH

**解决方案**:
1. 使用 **Anaconda Prompt** 而不是普通 CMD
2. 或者手动添加到 PATH：
   - 打开"环境变量"设置
   - 添加 Anaconda 安装路径（如 `C:\ProgramData\Anaconda3`）
   - 添加 Scripts 路径（如 `C:\ProgramData\Anaconda3\Scripts`）

### Q2: conda activate 不工作

**解决方案**:

在 Anaconda Prompt 中运行：
```bash
conda init cmd.exe
```

然后重启命令提示符。

### Q3: 环境激活失败

**检查步骤**:
```bash
# 1. 确认环境存在
conda env list

# 2. 查看环境中的包
conda list -n pointnet

# 3. 重新创建环境
conda env remove -n pointnet
conda env create -f environment.yml
```

### Q4: PyTorch CUDA 版本问题

**查看 CUDA 版本**:
```bash
nvidia-smi
```

**安装对应版本的 PyTorch**:

CUDA 11.8:
```bash
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

CUDA 12.1:
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

仅 CPU:
```bash
conda install pytorch cpuonly -c pytorch
```

### Q5: 批处理脚本在普通 CMD 中失败

**原因**: 普通 CMD 可能没有初始化 conda

**解决方案**:
1. 使用 **Anaconda Prompt**
2. 或在 CMD 中手动初始化：
   ```bash
   call C:\ProgramData\Anaconda3\Scripts\activate.bat
   conda activate pointnet
   ```

### Q6: 包冲突或版本问题

**清理并重建环境**:
```bash
# 删除旧环境
conda env remove -n pointnet

# 清理缓存
conda clean --all

# 重新创建
conda env create -f environment.yml
```

## 📊 Anaconda vs Pip 对比

| 特性 | Anaconda | Pip |
|------|----------|-----|
| 环境隔离 | ✅ 完整隔离 | ⚠️ 需配合 venv |
| 包管理 | ✅ 二进制包，速度快 | ⚠️ 编译慢 |
| 依赖管理 | ✅ 自动解决冲突 | ⚠️ 可能冲突 |
| CUDA 支持 | ✅ 自动配置 | ⚠️ 需手动选择 |
| 适合人群 | 科学计算、深度学习 | 通用 Python 开发 |

## 🔍 验证安装

### 检查 Conda 环境

```bash
conda activate pointnet
conda list
```

应该看到：
- pytorch
- numpy
- tqdm
- plyfile

### 检查 PyTorch

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 测试数据加载

```bash
python test_dataset.py
```

预期输出：
```
✓ Training dataset loaded successfully!
✓ Test dataset loaded successfully!
✓ All tests passed!
```

## 🎓 Anaconda Prompt 使用技巧

### 打开方式

1. **开始菜单**: 搜索 "Anaconda Prompt"
2. **快捷键**: `Win + R` → 输入 `cmd` → 在CMD中运行 `conda activate pointnet`
3. **右键菜单**: 在文件夹空白处按住 Shift + 右键 → "在此处打开 Anaconda Prompt"

### 设置默认激活环境

编辑 `~/.condarc` 或 `C:\Users\用户名\.condarc`:
```yaml
auto_activate_base: false
env_prompt: '({name}) '
```

### 添加到右键菜单

在 Anaconda Prompt 中运行：
```bash
conda install -c conda-forge conda-integration
```

## 📈 性能优化（Conda）

### 使用国内镜像加速

```bash
# 清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

# 中科大镜像
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
```

### 使用 mamba 加速（可选）

```bash
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

mamba 是 conda 的 C++ 重写版本，速度更快。

## 🎯 推荐工作流程

### 日常训练流程

```bash
# 1. 打开 Anaconda Prompt

# 2. 进入项目目录
cd C:\path\to\project

# 3. 激活环境
conda activate pointnet

# 4. 运行训练
cd pointnet.pytorch-master\utils
python train_classification.py --dataset ..\..\modelnet40_normal_resampled --dataset_type modelnet40 --nepoch 250

# 5. 训练完成后退出环境
conda deactivate
```

### 快速测试流程

```bash
# 在 Anaconda Prompt 中
cd C:\path\to\project
RUN_TRAINING_ANACONDA.bat
```

## 📦 环境管理最佳实践

1. **为每个项目创建独立环境**
   ```bash
   conda create -n project_name python=3.8
   ```

2. **使用 environment.yml 管理依赖**
   ```bash
   conda env export > environment.yml
   ```

3. **定期清理缓存**
   ```bash
   conda clean --all
   ```

4. **备份环境配置**
   ```bash
   conda list --export > requirements.txt
   ```

## ✅ 完整安装检查清单

在 Anaconda Prompt 中依次确认：

- [ ] `conda --version` - Anaconda 已安装
- [ ] `conda env list` - 能看到 pointnet 环境
- [ ] `conda activate pointnet` - 能激活环境
- [ ] `python --version` - Python 3.6+
- [ ] `python -c "import torch"` - PyTorch 已安装
- [ ] `python test_dataset.py` - 数据集加载正常
- [ ] 运行 `CHECK_ENVIRONMENT_ANACONDA.bat` 全部通过

全部通过？运行 `RUN_TRAINING_ANACONDA.bat` 开始训练！

## 🆘 需要帮助？

1. 查看 Conda 文档: https://docs.conda.io/
2. 查看 PyTorch 安装指南: https://pytorch.org/get-started/locally/
3. 检查 `README_ANACONDA.md` 获取详细信息
4. 在 Anaconda Prompt 中运行 `CHECK_ENVIRONMENT_ANACONDA.bat` 诊断问题

---

**专为 Anaconda 用户优化，开箱即用！** 🐍
