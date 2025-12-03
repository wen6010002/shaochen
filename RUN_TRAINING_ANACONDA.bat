@echo off
REM PointNet Classification Training Script for Windows with Anaconda
REM
REM This script automatically activates the conda environment and runs training
REM
REM Usage:
REM   - Quick test (2 epochs): RUN_TRAINING_ANACONDA.bat
REM   - Full training (250 epochs): RUN_TRAINING_ANACONDA.bat 250
REM   - Custom environment: RUN_TRAINING_ANACONDA.bat 10 myenv

echo ========================================
echo PointNet Training with Anaconda
echo ========================================
echo.

REM Set default values
set EPOCHS=2
set CONDA_ENV=pointnet

REM Parse arguments
if not "%1"=="" set EPOCHS=%1
if not "%2"=="" set CONDA_ENV=%2

echo Configuration:
echo - Conda Environment: %CONDA_ENV%
echo - Dataset: modelnet40_normal_resampled
echo - Epochs: %EPOCHS%
echo - Batch Size: 32
echo - Points per sample: 2500
echo.

REM Initialize conda for batch script
echo Initializing Anaconda...
call conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda is not found in PATH
    echo Please make sure Anaconda/Miniconda is installed and added to PATH
    echo.
    echo If Anaconda is installed, try running this in Anaconda Prompt instead
    pause
    exit /b 1
)

REM Activate conda environment
echo Activating conda environment: %CONDA_ENV%
call conda activate %CONDA_ENV%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda environment: %CONDA_ENV%
    echo.
    echo Please create the environment first:
    echo   conda env create -f environment.yml
    echo.
    echo Or activate manually in Anaconda Prompt and run:
    echo   cd pointnet.pytorch-master\utils
    echo   python train_classification.py --dataset ..\..\modelnet40_normal_resampled --dataset_type modelnet40 --nepoch %EPOCHS%
    pause
    exit /b 1
)

echo Environment activated successfully!
echo.

REM Add MKL library paths to PATH to fix DLL loading issues
echo Setting up MKL library paths...
set "PATH=%CONDA_PREFIX%\Library\bin;%PATH%"
set "PATH=%CONDA_PREFIX%\Library\mingw-w64\bin;%PATH%"
set "PATH=%CONDA_PREFIX%\bin;%PATH%"
echo [OK] Library paths configured
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
echo.

REM Deactivate environment
call conda deactivate

pause
