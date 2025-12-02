@echo off
REM PointNet Classification Training Script for Windows
REM
REM Usage:
REM   - Quick test (2 epochs): RUN_TRAINING_WINDOWS.bat
REM   - Full training (250 epochs): RUN_TRAINING_WINDOWS.bat 250

echo ========================================
echo PointNet Training on ModelNet40
echo ========================================
echo.

REM Set default epoch count
set EPOCHS=2
if not "%1"=="" set EPOCHS=%1

echo Configuration:
echo - Dataset: modelnet40_normal_resampled
echo - Epochs: %EPOCHS%
echo - Batch Size: 32
echo - Points per sample: 2500
echo.

REM Change to utils directory
cd pointnet.pytorch-master\utils

echo Starting training...
echo.

REM Run training
python train_classification.py --dataset ..\..\modelnet40_normal_resampled --dataset_type modelnet40 --nepoch %EPOCHS% --batchSize 32 --num_points 2500

echo.
echo ========================================
echo Training completed!
echo Models saved in: pointnet.pytorch-master\utils\cls\
echo ========================================

pause
