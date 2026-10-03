@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ==========================================
echo Starting Hear His Voice
echo ==========================================
echo.

REM ---------------------------------------------------------------------------
REM Activate the project Python virtual environment.
REM This ensures Hear His Voice always uses the project's dependencies regardless
REM of which Python environment is active in the terminal or system PATH.

if not exist "%~dp0.venv\Scripts\activate.bat" (
    echo.
    echo ERROR: Python virtual environment not found.
    echo Expected: %~dp0.venv
    echo Please run install_windows.bat first.
    exit /b 1
)

call "%~dp0.venv\Scripts\activate.bat"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate the Python virtual environment.
    exit /b 1
)

echo Python virtual environment activated.
echo.

set "OLLAMA_URL=http://localhost:11434"

echo Checking Ollama...

where ollama >nul 2>&1

if errorlevel 1 (
    echo.
    echo ERROR: Ollama is not installed.
    echo Please run install_windows.bat first.
    exit /b 1
)

echo Ollama detected.
echo.

echo Stopping existing Ollama processes...

taskkill /F /IM ollama.exe >nul 2>&1
taskkill /F /IM ollama_llama_server.exe >nul 2>&1
taskkill /F /IM llama-server.exe >nul 2>&1

timeout /t 3 /nobreak >nul

echo Existing Ollama processes stopped.
echo.

echo Starting Ollama on CPU...

set "OLLAMA_NUM_GPU=0"
set "OLLAMA_VULKAN=0"
set "OLLAMA_NO_CLOUD=1"

start "" /B cmd /c "ollama serve >nul 2>&1"

echo Ollama started.
echo.

echo Waiting for Ollama...

set "OLLAMA_READY=0"

for /L %%i in (1,1,30) do (

    curl -s --max-time 2 "%OLLAMA_URL%/api/tags" >nul 2>&1

    if not errorlevel 1 (
        set "OLLAMA_READY=1"
        goto :ollama_ready
    )

    timeout /t 1 /nobreak >nul
)

:ollama_ready

if "%OLLAMA_READY%"=="0" (
    echo.
    echo ERROR: Ollama failed to start.
    echo.
    exit /b 1
)

echo Ollama is ready.
echo.

echo Checking Ollama model...

ollama list | findstr /C:"qwen3:8b" >nul 2>&1

if errorlevel 1 (

    echo qwen3:8b not found.
    echo Pulling qwen3:8b...
    echo.

    ollama pull qwen3:8b

    if errorlevel 1 (
        echo.
        echo ERROR: Failed to pull qwen3:8b.
        exit /b 1
    )

) else (

    echo qwen3:8b detected.

)

echo.

echo Checking GPU memory...

where nvidia-smi >nul 2>&1

if not errorlevel 1 (
    nvidia-smi
    echo.
)

echo.

REM ---------------------------------------------------------------------------
REM SnapGenAI Chrome
REM
REM Start a dedicated visible Chrome profile with remote debugging enabled.
REM Selenium attaches to this browser instead of launching its own Chrome.
REM
REM The profile persists between runs so SnapGenAI login/session information
REM can be reused.

set "SNAPGENAI_DEBUG_HOST=127.0.0.1"
set "SNAPGENAI_DEBUG_PORT=9222"
set "SNAPGENAI_CHROME_PROFILE=%~dp0media\browser_profile\snapgenai"

echo ==========================================
echo Starting SnapGenAI Chrome
echo ==========================================
echo.

set "CHROME_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe"

if not exist "%CHROME_EXE%" set "CHROME_EXE=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"

if not exist "%CHROME_EXE%" (
    echo.
    echo ERROR: Google Chrome was not found.
    echo Please install Google Chrome.
    echo.
    exit /b 1
)

if not exist "%SNAPGENAI_CHROME_PROFILE%" mkdir "%SNAPGENAI_CHROME_PROFILE%"

echo Chrome executable:
echo %CHROME_EXE%
echo.

echo Chrome profile:
echo %SNAPGENAI_CHROME_PROFILE%
echo.

echo Remote debugging:
echo %SNAPGENAI_DEBUG_HOST%:%SNAPGENAI_DEBUG_PORT%
echo.

start "Hear His Voice - SnapGenAI Chrome" "%CHROME_EXE%" --remote-debugging-address=%SNAPGENAI_DEBUG_HOST% --remote-debugging-port=%SNAPGENAI_DEBUG_PORT% --user-data-dir="%SNAPGENAI_CHROME_PROFILE%" "https://snapgen.ai/"

echo SnapGenAI Chrome launched.
echo.
echo Selenium will attach to this browser on port %SNAPGENAI_DEBUG_PORT%.
echo.

timeout /t 3 /nobreak >nul

REM ---------------------------------------------------------------------------
REM Optional: auto-start a Cloudflare quick tunnel so Instagram uploads
REM work out of the box. Instagram's servers fetch the video from a
REM public URL, so publishing must be done through a publicly reachable
REM address. If cloudflared is not installed, Hear His Voice still starts
REM normally in local-only mode (YouTube uploads are unaffected).

set "SERVER_PORT=8000"
set "TUNNEL_LOG=%TEMP%\hhv_cloudflared.log"

set "CLOUDFLARED_EXE="

where cloudflared >nul 2>&1

if not errorlevel 1 (
    set "CLOUDFLARED_EXE=cloudflared"
    goto :cloudflared_found
)

if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\cloudflared.exe" (
    set "CLOUDFLARED_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Links\cloudflared.exe"
    goto :cloudflared_found
)

if exist "C:\Program Files (x86)\cloudflared\cloudflared.exe" (
    set "CLOUDFLARED_EXE=C:\Program Files (x86)\cloudflared\cloudflared.exe"
    goto :cloudflared_found
)

if exist "C:\Program Files\cloudflared\cloudflared.exe" (
    set "CLOUDFLARED_EXE=C:\Program Files\cloudflared\cloudflared.exe"
    goto :cloudflared_found
)

if exist "C:\Tools\cloudflared.exe" (
    set "CLOUDFLARED_EXE=C:\Tools\cloudflared.exe"
    goto :cloudflared_found
)

if exist "%USERPROFILE%\cloudflared.exe" (
    set "CLOUDFLARED_EXE=%USERPROFILE%\cloudflared.exe"
    goto :cloudflared_found
)

:cloudflared_found

if "%CLOUDFLARED_EXE%"=="" (

    echo WARNING: cloudflared not found - running local-only.
    echo Instagram publishing needs a public URL; install cloudflared
    echo ^(winget install Cloudflare.cloudflared^) and rerun this script,
    echo or expose the port another way ^(e.g. RunPod HTTP proxy^).
    echo.

    goto :start_server
)

echo Starting Cloudflare tunnel...

del "%TUNNEL_LOG%" >nul 2>&1

start "hhv-cloudflared" /MIN cmd /c ""%CLOUDFLARED_EXE%" tunnel --url http://localhost:%SERVER_PORT% > "%TUNNEL_LOG%" 2>&1"

echo Waiting for the public tunnel URL...

set "PUBLIC_URL="
set "TUNNEL_LINE_FILE=%TEMP%\hhv_tunnel_line.txt"

for /L %%i in (1,1,30) do (

    if "!PUBLIC_URL!"=="" (

        timeout /t 1 /nobreak >nul

        findstr /R /C:"https://[a-zA-Z0-9-]*\.trycloudflare\.com" "%TUNNEL_LOG%" > "%TUNNEL_LINE_FILE%" 2>nul

        set /p CANDIDATE_LINE=<"!TUNNEL_LINE_FILE!"

        if not "!CANDIDATE_LINE!"=="" (

            REM Strip anything before the URL, then cut at the first
            REM space so trailing log decoration does not leak in.

            set "CANDIDATE_LINE=!CANDIDATE_LINE:*https://=https://!"

            for /f "tokens=1 delims= " %%u in ("!CANDIDATE_LINE!") do (
                set "PUBLIC_URL=%%u"
            )

        )

    )

)

del "%TUNNEL_LINE_FILE%" >nul 2>&1

if "%PUBLIC_URL%"=="" (

    echo WARNING: Tunnel did not report a URL yet - continuing anyway.
    echo Check "%TUNNEL_LOG%" if Instagram publishing fails.
    echo.

) else (

    echo ==========================================================
    echo   Public URL:
    echo   %PUBLIC_URL%
    echo ==========================================================
    echo.

)

:start_server

REM Ollama is already running separately in CPU-only mode.
REM Clear the Ollama-specific environment variables before
REM launching Hear His Voice so the video pipeline can use its GPU.

set "OLLAMA_NUM_GPU="
set "OLLAMA_VULKAN="
set "OLLAMA_NO_CLOUD="

python -m web.server

set "EXIT_CODE=%errorlevel%"

REM Stop the tunnel started earlier (harmless if none was started).

taskkill /F /IM cloudflared.exe >nul 2>&1

echo.

if "%EXIT_CODE%"=="0" (

    echo ==========================================
    echo        Hear His Voice Complete
    echo ==========================================

) else (

    echo ==========================================
    echo        Hear His Voice Failed
    echo ==========================================
    echo.
    echo Exit code: %EXIT_CODE%

)

exit /b %EXIT_CODE%