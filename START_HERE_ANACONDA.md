# 开始训练 - 给你朋友的简单指南

> **他使用 Anaconda，这是最简单的步骤！**

## 🎯 只需 3 步！

### 第 1 步：打开 Anaconda Prompt

在 Windows 开始菜单搜索 **"Anaconda Prompt"**，打开它（不是普通的 CMD！）

### 第 2 步：进入项目文件夹

```bash
cd C:\路径\到\你的项目文件夹
```

替换成实际的路径，例如：
```bash
cd C:\Users\张三\Desktop\pointnet项目
```

### 第 3 步：创建环境并开始训练

```bash
# 一次性创建完整环境
conda env create -f environment.yml

# 等待安装完成后，激活环境
conda activate pointnet

# 进入项目目录安装 pointnet 包
cd pointnet.pytorch-master
pip install -e .
cd ..

# 开始训练！
RUN_TRAINING_ANACONDA.bat
```

**就这么简单！** 🎉

---

## 📊 训练过程

训练开始后会看到：

```
========================================
PointNet Training with Anaconda
========================================

Configuration:
- Conda Environment: pointnet
- Dataset: modelnet40_normal_resampled
- Epochs: 2
- Batch Size: 32

Initializing Anaconda...
Activating conda environment: pointnet
Environment activated successfully!

Starting training...

Using device: cuda  (或 cpu)
9843 2468
classes 40
[0: 0/307] train loss: 3.689120 accuracy: 0.031250
[0: 0/307] test loss: 3.682451 accuracy: 0.062500
...
```

**快速测试（2轮）**: 5-10 分钟（CPU）或 1 分钟（GPU）

---

## ⚡ 快速参考

### 再次运行训练

```bash
# 在 Anaconda Prompt 中
cd C:\路径\到\项目文件夹
RUN_TRAINING_ANACONDA.bat
```

### 修改训练轮数

```bash
# 训练 50 轮
RUN_TRAINING_ANACONDA.bat 50

# 完整训练 250 轮
RUN_TRAINING_ANACONDA.bat 250
```

### 检查环境是否正常

```bash
CHECK_ENVIRONMENT_ANACONDA.bat
```

### 常用 Conda 命令

```bash
# 查看所有环境
conda env list

# 激活 pointnet 环境
conda activate pointnet

# 退出环境
conda deactivate

# 删除环境（如果需要重新安装）
conda env remove -n pointnet
```

---

## ❓ 遇到问题？

### 问题 1：找不到 conda 命令

**解决**：确保在 **Anaconda Prompt** 中运行，不是普通 CMD

### 问题 2：环境创建失败

```bash
# 清理并重试
conda clean --all
conda env create -f environment.yml
```

### 问题 3：找不到数据集

确认：
- `modelnet40_normal_resampled` 文件夹在项目根目录
- 文件夹里有 40 个子文件夹（airplane, bathtub...）

### 问题 4：训练太慢

正常现象！如果是 CPU 训练会很慢。建议：
- 先用 2 轮测试看能不能跑
- 有 GPU 的话会快很多

---

## 📁 训练完成后

模型保存在：
```
pointnet.pytorch-master\utils\cls\cls_model_0.pth
pointnet.pytorch-master\utils\cls\cls_model_1.pth
...
```

每个 `.pth` 文件是一个训练好的模型。

---

## 🎓 需要详细说明？

查看完整文档：
- **快速上手**: QUICKSTART_ANACONDA.md
- **详细手册**: README_ANACONDA.md
- **所有文档**: README.md

---

**就这么简单！创建环境 → 运行脚本 → 开始训练！** 🚀

有问题随时查看文档或运行 `CHECK_ENVIRONMENT_ANACONDA.bat` 诊断问题。
