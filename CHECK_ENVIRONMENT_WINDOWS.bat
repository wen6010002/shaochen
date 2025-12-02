@echo off
REM Environment Check Script for PointNet Training
REM This script verifies that your system is ready to run PointNet training

echo ========================================
echo PointNet Training Environment Check
echo ========================================
echo.

set ERRORS=0

REM Check 1: Python installation
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo     [FAIL] Python is not installed or not in PATH
    set /a ERRORS+=1
) else (
    python --version
    echo     [PASS] Python is installed
)
echo.

REM Check 2: PyTorch
echo [2/7] Checking PyTorch...
python -c "import torch; print('     [PASS] PyTorch', torch.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo     [FAIL] PyTorch is not installed
    echo     Run: pip install torch
    set /a ERRORS+=1
)
echo.

REM Check 3: NumPy
echo [3/7] Checking NumPy...
python -c "import numpy; print('     [PASS] NumPy', numpy.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo     [FAIL] NumPy is not installed
    echo     Run: pip install numpy
    set /a ERRORS+=1
)
echo.

REM Check 4: tqdm
echo [4/7] Checking tqdm...
python -c "import tqdm; print('     [PASS] tqdm installed')" 2>nul
if %errorlevel% neq 0 (
    echo     [FAIL] tqdm is not installed
    echo     Run: pip install tqdm
    set /a ERRORS+=1
)
echo.

REM Check 5: plyfile
echo [5/7] Checking plyfile...
python -c "import plyfile; print('     [PASS] plyfile installed')" 2>nul
if %errorlevel% neq 0 (
    echo     [FAIL] plyfile is not installed
    echo     Run: pip install plyfile
    set /a ERRORS+=1
)
echo.

REM Check 6: Dataset directory
echo [6/7] Checking dataset directory...
if exist "modelnet40_normal_resampled\" (
    echo     [PASS] Dataset directory found
) else (
    echo     [FAIL] modelnet40_normal_resampled directory not found
    echo     Make sure the dataset is in the correct location
    set /a ERRORS+=1
)
echo.

REM Check 7: Dataset files
echo [7/7] Checking dataset files...
if exist "modelnet40_normal_resampled\trainval.txt" (
    echo     [PASS] trainval.txt found
) else (
    echo     [FAIL] trainval.txt not found
    set /a ERRORS+=1
)
if exist "modelnet40_normal_resampled\test.txt" (
    echo     [PASS] test.txt found
) else (
    echo     [FAIL] test.txt not found
    set /a ERRORS+=1
)
echo.

REM GPU Check (optional)
echo [BONUS] Checking CUDA availability...
python -c "import torch; print('     GPU Available:', torch.cuda.is_available()); print('     Device:', 'CUDA' if torch.cuda.is_available() else 'CPU (training will be slower)')" 2>nul
echo.

REM Summary
echo ========================================
if %ERRORS%==0 (
    echo [SUCCESS] All checks passed!
    echo You are ready to start training.
    echo.
    echo Run training with:
    echo   RUN_TRAINING_WINDOWS.bat
) else (
    echo [FAILED] Found %ERRORS% error(s)
    echo Please fix the errors above before training.
    echo.
    echo To install missing dependencies:
    echo   pip install torch tqdm plyfile numpy
    echo   cd pointnet.pytorch-master
    echo   pip install -e .
)
echo ========================================

pause
