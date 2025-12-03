@echo off
REM ========================================
REM Quick Test Script - Verify Fix
REM ========================================

echo ========================================
echo Quick Test - Verify MKL Fix
echo ========================================
echo.

set CONDA_ENV=pointnet

call conda activate %CONDA_ENV%
if %errorlevel% neq 0 (
    echo [ERROR] Cannot activate environment: %CONDA_ENV%
    echo Please create the environment first
    pause
    exit /b 1
)

echo [INFO] Testing environment: %CONDA_ENV%
echo.

echo ========================================
echo Test 1: Python Version
echo ========================================
python --version
echo.

echo ========================================
echo Test 2: Import NumPy
echo ========================================
python -c "import numpy as np; print('NumPy version:', np.__version__); print('[OK] NumPy works!')"
if %errorlevel% neq 0 (
    echo [FAILED] NumPy import failed!
    goto FAILED
)
echo.

echo ========================================
echo Test 3: Import PyTorch
echo ========================================
python -c "import torch; print('PyTorch version:', torch.__version__); print('[OK] PyTorch works!')"
if %errorlevel% neq 0 (
    echo [FAILED] PyTorch import failed!
    goto FAILED
)
echo.

echo ========================================
echo Test 4: Import Training Dependencies
echo ========================================
python -c "import numpy; import torch; from tqdm import tqdm; import plyfile; print('[OK] All dependencies work!')"
if %errorlevel% neq 0 (
    echo [FAILED] Some dependencies missing!
    goto FAILED
)
echo.

echo ========================================
echo Test 5: Quick Computation Test
echo ========================================
python -c "import numpy as np; import torch; x = np.random.rand(100, 100); y = torch.randn(100, 100); print('[OK] Computations work!')"
if %errorlevel% neq 0 (
    echo [FAILED] Computation test failed!
    goto FAILED
)
echo.

echo ========================================
echo ALL TESTS PASSED! ✓
echo ========================================
echo.
echo Your environment is ready!
echo You can now run: RUN_TRAINING_ANACONDA.bat
echo.
goto END

:FAILED
echo.
echo ========================================
echo TESTS FAILED! ✗
echo ========================================
echo.
echo Please try one of these fixes:
echo 1. FIX_MKL_ALTERNATIVE.bat (recommended)
echo 2. FIX_MKL_ULTIMATE.bat (tries all methods)
echo 3. Read MKL_FIX_README.md for detailed instructions
echo.

:END
call conda deactivate
pause
