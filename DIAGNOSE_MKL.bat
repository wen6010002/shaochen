@echo off
REM ========================================
REM MKL Problem Diagnostic Script
REM ========================================

echo ========================================
echo MKL Library Diagnostic Tool
echo ========================================
echo.

set CONDA_ENV=pointnet

REM Activate environment
call conda activate %CONDA_ENV%
if %errorlevel% neq 0 (
    echo [ERROR] Cannot activate environment: %CONDA_ENV%
    pause
    exit /b 1
)

echo [INFO] Environment: %CONDA_ENV%
echo [INFO] Python Version:
python --version
echo.

echo ========================================
echo 1. Checking Conda Package List
echo ========================================
echo [Checking MKL-related packages...]
conda list | findstr /i "mkl"
echo.
conda list | findstr /i "numpy"
echo.
conda list | findstr /i "torch"
echo.

echo ========================================
echo 2. Locating MKL DLL Files
echo ========================================
echo [Searching for mkl_intel_thread.dll...]
where /R "%CONDA_PREFIX%" mkl_intel_thread.dll 2>nul
if %errorlevel% equ 0 (
    echo [OK] Found mkl_intel_thread.dll
) else (
    echo [ERROR] mkl_intel_thread.dll NOT FOUND!
)
echo.

echo [Searching for mkl_core.dll...]
where /R "%CONDA_PREFIX%" mkl_core.dll 2>nul
if %errorlevel% equ 0 (
    echo [OK] Found mkl_core.dll
) else (
    echo [ERROR] mkl_core.dll NOT FOUND!
)
echo.

echo ========================================
echo 3. Checking Library Paths
echo ========================================
echo [Conda Prefix:]
echo %CONDA_PREFIX%
echo.
echo [Library Bin Path:]
if exist "%CONDA_PREFIX%\Library\bin" (
    echo %CONDA_PREFIX%\Library\bin
    echo [OK] Library\bin directory exists
    dir "%CONDA_PREFIX%\Library\bin\mkl*.dll" 2>nul | findstr /i "mkl"
) else (
    echo [ERROR] Library\bin directory NOT FOUND!
)
echo.

echo ========================================
echo 4. Testing Python Imports
echo ========================================
echo [Testing NumPy import...]
python -c "import numpy as np; print('NumPy version:', np.__version__); print('NumPy imported successfully')" 2>&1
if %errorlevel% equ 0 (
    echo [OK] NumPy works
) else (
    echo [ERROR] NumPy import failed!
)
echo.

echo [Testing PyTorch import...]
python -c "import torch; print('PyTorch version:', torch.__version__); print('PyTorch imported successfully')" 2>&1
if %errorlevel% equ 0 (
    echo [OK] PyTorch works
) else (
    echo [ERROR] PyTorch import failed!
)
echo.

echo ========================================
echo 5. System Information
echo ========================================
echo [Operating System:]
ver
echo.
echo [Current PATH:]
echo %PATH%
echo.

echo ========================================
echo Diagnostic Complete
echo ========================================
echo.
echo Please save this output and check:
echo 1. Whether mkl_intel_thread.dll was found
echo 2. If Library\bin directory exists
echo 3. If NumPy/PyTorch import successfully
echo.

pause
