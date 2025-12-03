@echo off
REM ========================================
REM Alternative Fix: Remove MKL and Use OpenBLAS
REM ========================================
REM
REM This script removes MKL libraries and uses OpenBLAS instead
REM This is slower but avoids the DLL loading issues

echo ========================================
echo MKL Alternative Fix - OpenBLAS Method
echo ========================================
echo.
echo WARNING: This will remove MKL libraries and use OpenBLAS instead.
echo This is slower but more stable on some Windows systems.
echo.
echo Press Ctrl+C to cancel, or
pause

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
    pause
    exit /b 1
)
echo [OK] Environment activated
echo.

echo Step 2: Installing nomkl package...
call conda install -y nomkl
echo [OK] nomkl installed
echo.

echo Step 3: Removing MKL packages...
call conda remove -y mkl mkl-service --force
echo [OK] MKL packages removed
echo.

echo Step 4: Reinstalling NumPy with OpenBLAS...
call conda install -y numpy scipy -c conda-forge
echo [OK] NumPy reinstalled with OpenBLAS
echo.

echo Step 5: Reinstalling PyTorch (CPU version)...
call conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
echo [OK] PyTorch reinstalled
echo.

echo Step 6: Testing installation...
python -c "import numpy as np; import torch; print('NumPy version:', np.__version__); print('PyTorch version:', torch.__version__); print('Test successful!')"
if %errorlevel% equ 0 (
    echo [OK] All libraries working correctly!
) else (
    echo [WARNING] Import test failed, but you can try running training anyway
)
echo.

echo ========================================
echo Alternative Fix Applied!
echo ========================================
echo.
echo MKL has been removed and replaced with OpenBLAS.
echo You can now run: RUN_TRAINING_ANACONDA.bat
echo.
echo Note: Training may be slightly slower but should work without errors.
echo.

call conda deactivate

pause
