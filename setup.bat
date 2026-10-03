@echo off
setlocal

echo ==========================================
echo       Hear His Voice Windows Installer
echo ==========================================

echo.
echo Checking Python...

python --version >nul 2>&1

if %errorlevel% neq 0 (
    echo ERROR: Python not found.
    echo Please install Python 3.12 first.
    pause
    exit /b 1
)

python --version

echo.
echo Creating virtual environment...

if not exist ".venv\Scripts\activate.bat" (
    python -m venv .venv

    if %errorlevel% neq 0 (
        echo ERROR: Failed creating virtual environment.
        pause
        exit /b 1
    )
)

echo Virtual environment ready.

echo.
echo Activating virtual environment...

call .venv\Scripts\activate.bat

echo.
echo Upgrading pip...

python -m pip install --upgrade pip

if %errorlevel% neq 0 (
    echo ERROR: Failed upgrading pip.
    pause
    exit /b 1
)

echo.
echo Checking NVIDIA GPU...

nvidia-smi >nul 2>&1

if %errorlevel% equ 0 (

    echo NVIDIA GPU detected.
    echo Installing CUDA PyTorch...

    python -m pip install --upgrade ^
        torch ^
        torchvision ^
        torchaudio ^
        --index-url https://download.pytorch.org/whl/cu128

) else (

    echo No NVIDIA GPU detected.
    echo Installing CPU PyTorch...

    python -m pip install --upgrade ^
        torch ^
        torchvision ^
        torchaudio

)

if %errorlevel% neq 0 (
    echo ERROR: Failed installing PyTorch.
    pause
    exit /b 1
)

echo.
echo Installing Hear His Voice dependencies...

python -m pip install --upgrade -r requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: Failed installing dependencies.
    pause
    exit /b 1
)

echo.
echo Checking FFmpeg...

ffmpeg -version >nul 2>&1

if %errorlevel% neq 0 (

    echo FFmpeg not found.
    echo Installing FFmpeg using winget...

    winget install Gyan.FFmpeg ^
        --accept-package-agreements ^
        --accept-source-agreements

    if %errorlevel% neq 0 (
        echo ERROR: Failed installing FFmpeg.
        pause
        exit /b 1
    )

) else (

    echo FFmpeg detected.

)

echo.
echo Checking Ollama...

where ollama >nul 2>&1

if %errorlevel% neq 0 (

    echo Ollama not found.

    echo.
    echo Please install Ollama for Windows from:
    echo https://ollama.com/download/windows

    echo.
    echo After installing Ollama, run this installer again.

    pause
    exit /b 1

) else (

    echo Ollama detected.

)

echo.
echo Stopping existing Ollama processes...

taskkill /F /IM ollama.exe >nul 2>&1
taskkill /F /IM "ollama app.exe" >nul 2>&1
taskkill /F /IM "ollama_llama_server.exe" >nul 2>&1

timeout /t 2 /nobreak >nul

echo.
echo Starting Ollama in CPU-only mode...

set "OLLAMA_CPU_SCRIPT=%TEMP%\hhv_ollama_cpu.bat"

(
    echo @echo off
    echo set "CUDA_VISIBLE_DEVICES="
    echo set "NVIDIA_VISIBLE_DEVICES="
    echo set "OLLAMA_VULKAN=0"
    echo set "OLLAMA_NUM_GPU=0"
    echo ollama serve ^> "%CD%\ollama.log" 2^>^&1
) > "%OLLAMA_CPU_SCRIPT%"

start "Hear His Voice Ollama" /min cmd /c call "%OLLAMA_CPU_SCRIPT%"

echo.
echo Waiting for Ollama...

set "OLLAMA_READY=0"

for /L %%i in (1,1,30) do (

    curl -s --max-time 2 http://localhost:11434/api/tags >nul 2>&1

    if not errorlevel 1 (
        set "OLLAMA_READY=1"
        goto :ollama_ready
    )

    timeout /t 1 /nobreak >nul

)

:ollama_ready

if "%OLLAMA_READY%" neq "1" (

    echo.
    echo ERROR: Ollama failed to start.

    echo.
    echo Ollama log:
    echo ------------------------------------------

    if exist ollama.log (
        type ollama.log
    ) else (
        echo Ollama log not found.
    )

    echo ------------------------------------------

    if exist "%OLLAMA_CPU_SCRIPT%" (
        del "%OLLAMA_CPU_SCRIPT%" >nul 2>&1
    )

    pause
    exit /b 1

)

echo Ollama is ready.

if exist "%OLLAMA_CPU_SCRIPT%" (
    del "%OLLAMA_CPU_SCRIPT%" >nul 2>&1
)

echo.
echo Checking Qwen model...

ollama list | findstr /C:"qwen3:8b" >nul

if %errorlevel% neq 0 (

    echo qwen3:8b not found.
    echo Downloading qwen3:8b...

    ollama pull qwen3:8b

    if %errorlevel% neq 0 (
        echo ERROR: Failed to pull qwen3:8b.
        pause
        exit /b 1
    )

) else (

    echo qwen3:8b already installed.

)

echo.
echo Running hardware verification...

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo Checking GPU memory...

nvidia-smi >nul 2>&1

if %errorlevel% equ 0 (
    nvidia-smi
)

echo.
echo ==========================================
echo       Installation Complete
echo ==========================================

echo.
echo Run Hear His Voice with:
echo.
echo run_windows.bat
echo.

pause