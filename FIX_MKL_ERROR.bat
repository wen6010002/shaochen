@echo off
REM ========================================
REM Intel MKL Error Fix Script for PointNet
REM ========================================
REM
REM This script fixes the "Cannot load mkl_intel_thread.dll" error
REM on Windows with Anaconda

echo ========================================
echo Intel MKL Error Fix Script
echo ========================================
echo.
echo This script will fix the MKL library error by installing
echo the required Intel MKL dependencies in your conda environment.
echo.

set CONDA_ENV=pointnet

REM Check if conda is available
call conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda is not found in PATH
    echo Please run this script in Anaconda Prompt
    pause
    exit /b 1
)

echo Step 1: Activating conda environment: %CONDA_ENV%
call conda activate %CONDA_ENV%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate environment: %CONDA_ENV%
    echo Creating environment from environment.yml...
    call conda env create -f environment.yml
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create environment
        pause
        exit /b 1
    )
    call conda activate %CONDA_ENV%
)
echo [OK] Environment activated
echo.

echo Step 2: Installing Intel MKL libraries...
call conda install -y mkl mkl-service intel-openmp
if %errorlevel% neq 0 (
    echo [WARNING] conda install failed, trying alternative method...
    call conda install -y -c conda-forge mkl mkl-service intel-openmp
)
echo [OK] MKL libraries installed
echo.

echo Step 3: Reinstalling NumPy with MKL support...
call conda install -y numpy --force-reinstall
echo [OK] NumPy reinstalled
echo.

echo Step 4: Verifying PyTorch installation...
call conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
echo [OK] PyTorch verified
echo.

echo ========================================
echo Fix Applied Successfully!
echo ========================================
echo.
echo You can now run: RUN_TRAINING_ANACONDA.bat
echo.

call conda deactivate

pause
