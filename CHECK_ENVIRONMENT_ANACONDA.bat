@echo off
REM Environment Check Script for PointNet Training with Anaconda
REM This script verifies that your Anaconda environment is ready

echo ========================================
echo PointNet Anaconda Environment Check
echo ========================================
echo.

set ERRORS=0
set CONDA_ENV=pointnet

REM Check 1: Conda installation
echo [1/9] Checking Anaconda/Miniconda installation...
call conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo     [FAIL] Conda is not found
    echo     Please install Anaconda or Miniconda
    echo     Download from: https://www.anaconda.com/download
    set /a ERRORS+=1
) else (
    call conda --version
    echo     [PASS] Conda is installed
)
echo.

REM Check 2: Conda environment exists
echo [2/9] Checking if '%CONDA_ENV%' environment exists...
call conda env list | findstr /C:"%CONDA_ENV%" >nul 2>&1
if %errorlevel% neq 0 (
    echo     [WARN] Environment '%CONDA_ENV%' not found
    echo     You need to create it with:
    echo       conda env create -f environment.yml
    echo.
    set /a ERRORS+=1
) else (
    echo     [PASS] Environment '%CONDA_ENV%' exists
)
echo.

REM Try to activate environment for further checks
echo [3/9] Activating environment '%CONDA_ENV%'...
call conda activate %CONDA_ENV% >nul 2>&1
if %errorlevel% neq 0 (
    echo     [FAIL] Cannot activate environment
    echo     Please create the environment first:
    echo       conda env create -f environment.yml
    set /a ERRORS+=1
    goto :skip_env_checks
) else (
    echo     [PASS] Environment activated
)
echo.

REM Check 3: Python
echo [4/9] Checking Python in conda environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo     [FAIL] Python not found in environment
    set /a ERRORS+=1
) else (
    python --version
    echo     [PASS] Python is available
)
echo.

REM Check 4: PyTorch
echo [5/9] Checking PyTorch...
python -c "import torch; print('     [PASS] PyTorch', torch.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo     [FAIL] PyTorch is not installed in conda environment
    echo     Run: conda install pytorch -c pytorch
    set /a ERRORS+=1
)
echo.

REM Check 5: NumPy
echo [6/9] Checking NumPy...
python -c "import numpy; print('     [PASS] NumPy', numpy.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo     [FAIL] NumPy is not installed
    echo     Run: conda install numpy
    set /a ERRORS+=1
)
echo.

REM Check 6: tqdm
echo [7/9] Checking tqdm...
python -c "import tqdm; print('     [PASS] tqdm installed')" 2>nul
if %errorlevel% neq 0 (
    echo     [FAIL] tqdm is not installed
    echo     Run: conda install tqdm
    set /a ERRORS+=1
)
echo.

REM Check 7: plyfile
echo [8/9] Checking plyfile...
python -c "import plyfile; print('     [PASS] plyfile installed')" 2>nul
if %errorlevel% neq 0 (
    echo     [FAIL] plyfile is not installed
    echo     Run: pip install plyfile
    set /a ERRORS+=1
)
echo.

REM GPU Check
echo [9/9] Checking CUDA availability...
python -c "import torch; print('     GPU Available:', torch.cuda.is_available()); print('     Device:', 'CUDA' if torch.cuda.is_available() else 'CPU (training will be slower)')" 2>nul
echo.

call conda deactivate

:skip_env_checks

REM Check Dataset
echo [BONUS 1] Checking dataset directory...
if exist "modelnet40_normal_resampled\" (
    echo     [PASS] Dataset directory found
) else (
    echo     [FAIL] modelnet40_normal_resampled directory not found
    set /a ERRORS+=1
)
echo.

REM Check dataset files
echo [BONUS 2] Checking dataset files...
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

REM Summary
echo ========================================
if %ERRORS%==0 (
    echo [SUCCESS] All checks passed!
    echo You are ready to start training.
    echo.
    echo Run training with:
    echo   RUN_TRAINING_ANACONDA.bat
    echo.
    echo Or in Anaconda Prompt:
    echo   conda activate %CONDA_ENV%
    echo   cd pointnet.pytorch-master\utils
    echo   python train_classification.py --dataset ..\..\modelnet40_normal_resampled --dataset_type modelnet40 --nepoch 2
) else (
    echo [FAILED] Found %ERRORS% error(s)
    echo.
    if not exist "modelnet40_normal_resampled\" (
        echo Please ensure dataset is in the correct location
    )
    echo.
    echo To create the conda environment:
    echo   conda env create -f environment.yml
    echo.
    echo To install missing dependencies:
    echo   conda activate %CONDA_ENV%
    echo   conda install pytorch numpy tqdm -c pytorch
    echo   pip install plyfile
    echo   cd pointnet.pytorch-master
    echo   pip install -e .
)
echo ========================================

pause
