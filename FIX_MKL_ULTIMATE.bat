@echo off
REM ========================================
REM Ultimate MKL Fix Script - All Methods
REM ========================================

echo ========================================
echo Ultimate MKL Fix Script
echo ========================================
echo.
echo This script will try multiple methods to fix the MKL DLL error.
echo.
echo Your system info:
echo - Anaconda: 22.9.0
echo - Python: 3.8.20
echo - Windows: 11
echo.
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

echo.
echo ========================================
echo METHOD 1: Reinstall MKL with Dependencies
echo ========================================
echo.

call conda activate %CONDA_ENV%
if %errorlevel% neq 0 (
    echo [ERROR] Cannot activate environment
    goto METHOD_2
)

echo Installing Intel MKL and dependencies...
call conda install -y mkl=2021.4 mkl-service intel-openmp -c conda-forge
call conda install -y numpy=1.23.5 --force-reinstall
call conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

echo Testing...
python -c "import numpy; import torch; print('Method 1: Success')" 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Method 1 worked!
    goto SUCCESS
)

echo [FAILED] Method 1 didn't work, trying Method 2...
echo.

:METHOD_2
echo ========================================
echo METHOD 2: Use Specific MKL Version
echo ========================================
echo.

call conda activate %CONDA_ENV%

echo Installing specific MKL version known to work on Windows 11...
call conda install -y mkl=2020.4 mkl-service=2.3.0 -c anaconda
call conda install -y numpy=1.19.5 --force-reinstall
call conda install -y pytorch=1.10.0 torchvision torchaudio cpuonly -c pytorch

echo Testing...
python -c "import numpy; import torch; print('Method 2: Success')" 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Method 2 worked!
    goto SUCCESS
)

echo [FAILED] Method 2 didn't work, trying Method 3...
echo.

:METHOD_3
echo ========================================
echo METHOD 3: Remove MKL, Use OpenBLAS
echo ========================================
echo.

call conda activate %CONDA_ENV%

echo Removing MKL and switching to OpenBLAS...
call conda install -y nomkl
call conda remove -y mkl mkl-service --force
call conda install -y numpy scipy -c conda-forge
call conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

echo Testing...
python -c "import numpy; import torch; print('Method 3: Success')" 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Method 3 worked!
    goto SUCCESS
)

echo [FAILED] Method 3 didn't work, trying Method 4...
echo.

:METHOD_4
echo ========================================
echo METHOD 4: Complete Environment Rebuild
echo ========================================
echo.

echo This will delete and recreate your environment.
echo Press Ctrl+C to cancel, or
pause

call conda deactivate
call conda env remove -n %CONDA_ENV% -y

echo Creating new environment with minimal dependencies...
call conda create -n %CONDA_ENV% python=3.8 -y
call conda activate %CONDA_ENV%

call conda install -y nomkl numpy scipy -c conda-forge
call conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
call conda install -y tqdm -c conda-forge
call pip install plyfile

echo Testing...
python -c "import numpy; import torch; print('Method 4: Success')" 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Method 4 worked!
    goto SUCCESS
)

echo.
echo ========================================
echo ALL METHODS FAILED
echo ========================================
echo.
echo Please try the following:
echo 1. Update Anaconda: conda update conda
echo 2. Install Visual C++ Redistributable from Microsoft
echo 3. Check Windows Updates
echo 4. Run DIAGNOSE_MKL.bat and share the output
echo.
goto END

:SUCCESS
echo.
echo ========================================
echo FIX SUCCESSFUL!
echo ========================================
echo.
echo The MKL error should now be fixed.
echo You can now run: RUN_TRAINING_ANACONDA.bat
echo.

:END
call conda deactivate
pause
