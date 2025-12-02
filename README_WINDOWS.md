# PointNet 训练指南 - Windows 版本

## 📁 目录结构

确保你的文件结构如下：

```
项目根目录/
├── pointnet.pytorch-master/          # PointNet 项目代码
│   ├── pointnet/                     # 核心代码
│   │   ├── dataset.py               # 数据集加载（已修改）
│   │   └── model.py                 # 模型定义
│   └── utils/
│       └── train_classification.py  # 训练脚本（已修改）
├── modelnet40_normal_resampled/      # ModelNet40 数据集
│   ├── airplane/                     # 各类别文件夹
│   ├── bathtub/
│   ├── ... (共40个类别)
│   ├── trainval.txt                  # 训练集文件列表
│   └── test.txt                      # 测试集文件列表
├── RUN_TRAINING_WINDOWS.bat          # Windows 训练脚本
└── README_WINDOWS.md                 # 本文档
```

## 🔧 已修改的内容

### 1. 数据集加载 (pointnet/dataset.py)

**修改位置**: 第 167-179 行

**修改原因**:
- 原代码使用 `.ply` 格式，但 ModelNet40 数据集使用 `.txt` 格式
- 原代码路径处理不兼容 Windows 系统
- 文件名解析逻辑需要适配新的数据集格式

**修改内容**:
```python
def __getitem__(self, index):
    fn = self.fns[index]
    # 从文件名提取类别（例如 'airplane_0001' -> 'airplane'）
    basename = os.path.basename(fn)
    class_name = '_'.join(basename.split('_')[:-1])
    cls = self.cat[class_name]

    # 构建文件路径（Windows 兼容）
    file_path = os.path.join(self.root, class_name, fn)
    if not os.path.exists(file_path):
        file_path = file_path + '.txt'

    # 读取 .txt 格式数据（x,y,z,nx,ny,nz）
    if file_path.endswith('.txt'):
        point_set = np.loadtxt(file_path, delimiter=',').astype(np.float32)
        pts = point_set[:, 0:3]  # 只取 x,y,z 坐标
```

### 2. 训练脚本 (utils/train_classification.py)

**修改位置**: 第 94-152 行

**修改原因**:
- 原代码强制使用 CUDA，Windows 用户可能没有 NVIDIA GPU
- 需要自动检测并适配 CPU/GPU

**修改内容**:
```python
# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
classifier.to(device)

# 所有数据传输改为使用 device
points, target = points.to(device), target.to(device)
```

## 🚀 安装步骤

### 1. 安装 Python 依赖

打开 **命令提示符 (CMD)** 或 **PowerShell**，执行：

```bash
# 进入项目目录
cd 你的项目路径

# 进入 pointnet 项目
cd pointnet.pytorch-master

# 安装依赖
pip install torch tqdm plyfile numpy

# 安装 pointnet 包
pip install -e .
```

如果遇到 SSL 证书错误，使用：
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org torch tqdm plyfile numpy
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e .
```

## 🎯 运行训练

### 方法 1: 使用批处理脚本（推荐）

**快速测试（2 轮）：**
```bash
双击运行: RUN_TRAINING_WINDOWS.bat
```

**完整训练（250 轮）：**
```bash
RUN_TRAINING_WINDOWS.bat 250
```

**自定义轮数：**
```bash
RUN_TRAINING_WINDOWS.bat 50
```

### 方法 2: 手动运行

打开 CMD，执行：

```bash
cd pointnet.pytorch-master\utils

python train_classification.py ^
  --dataset ..\..\modelnet40_normal_resampled ^
  --dataset_type modelnet40 ^
  --nepoch 2 ^
  --batchSize 32 ^
  --num_points 2500
```

**注意**: Windows CMD 使用 `^` 作为行连接符

### 方法 3: PowerShell 运行

```powershell
cd pointnet.pytorch-master\utils

python train_classification.py `
  --dataset ..\..\modelnet40_normal_resampled `
  --dataset_type modelnet40 `
  --nepoch 2 `
  --batchSize 32 `
  --num_points 2500
```

**注意**: PowerShell 使用反引号 `` ` `` 作为行连接符

## 📊 训练参数说明

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--dataset` | 数据集路径 | 必填 | `..\..\modelnet40_normal_resampled` |
| `--dataset_type` | 数据集类型 | shapenet | `modelnet40` |
| `--nepoch` | 训练轮数 | 250 | 测试:2-10, 正式:250 |
| `--batchSize` | 批次大小 | 32 | 32 (GPU) / 16 (CPU) |
| `--num_points` | 点云采样数 | 2500 | 2500 |
| `--feature_transform` | 使用特征变换 | False | 可选 |
| `--workers` | 数据加载线程 | 4 | Windows 建议 0-2 |

## 💾 输出文件

训练过程中会自动创建 `cls/` 目录并保存模型：

```
pointnet.pytorch-master\utils\cls\
├── cls_model_0.pth      # 第 0 轮模型
├── cls_model_1.pth      # 第 1 轮模型
└── ...
```

## 🖥️ 预期输出

```
Using device: cuda  (如果有GPU)
或
Using device: cpu   (如果无GPU)

9843 2468  (训练集和测试集大小)
classes 40  (类别数)

[0: 0/307] train loss: 3.689120 accuracy: 0.031250
[0: 0/307] test loss: 3.682451 accuracy: 0.062500
[0: 10/307] train loss: 3.645123 accuracy: 0.093750
...
```

## ⚙️ Windows 特殊配置

### 1. 数据加载线程数

如果遇到多进程错误，修改 `train_classification.py`:

```python
# 原始代码
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))  # Windows 可能出错

# 修改为
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=0)  # Windows 使用 0
```

### 2. 路径问题

所有路径已使用 `os.path.join()` 处理，自动适配 Windows 反斜杠 `\`。

### 3. 内存不足

如果训练时内存不足，减小参数：
```bash
--batchSize 16 --num_points 1024
```

## 🐛 常见问题

### 问题 1: `FileNotFoundError`

**原因**: 数据集路径不正确

**解决**:
- 确认 `modelnet40_normal_resampled` 文件夹在正确位置
- 检查是否有 `trainval.txt` 和 `test.txt` 文件

### 问题 2: CUDA 相关错误

**解决**: 代码已自动适配，会在无 GPU 时使用 CPU

### 问题 3: 多进程加载错误

**解决**: 添加参数 `--workers 0`

### 问题 4: 训练速度慢

**原因**: CPU 训练比 GPU 慢很多

**建议**:
- 减少 epoch: `--nepoch 10`
- 减少 batch size: `--batchSize 8`
- 使用带 CUDA 的 GPU

## 📈 性能参考

### GPU (NVIDIA RTX 3080)
- 训练速度: ~100 samples/sec
- 每个 epoch: ~3-5 分钟
- 250 epochs: ~15-20 小时

### CPU (Intel i7)
- 训练速度: ~10-20 samples/sec
- 每个 epoch: ~30-60 分钟
- 250 epochs: ~5-10 天

**建议**: 如果使用 CPU，先用少量 epoch (10-20) 测试效果。

## ✅ 验证训练结果

训练完成后，最后会显示：

```
final accuracy 0.XXXX
```

预期准确率：
- 无特征变换: ~86-87%
- 有特征变换: ~87-88%

## 📝 修改总结

1. ✅ 数据加载适配 `.txt` 格式
2. ✅ 路径处理兼容 Windows
3. ✅ 自动检测 CPU/GPU
4. ✅ 类别名提取逻辑修正
5. ✅ 创建必要的文件列表（trainval.txt, test.txt）

所有修改已完成，可以直接在 Windows 上运行！
