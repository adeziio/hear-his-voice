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

REM ---------------------------------------------------------------------------
REM Ollama
REM
REM Ollama is a single shared server on a fixed port, so several projects
REM point at the same instance. It must therefore be REUSED, never killed:
REM stopping it would abort any generation running in another project.
REM Only start it when nothing answers yet, and allow more than one
REM simultaneous request so parallel projects do not queue behind
REM each other.
REM
REM NOTE ON GPU: this project normally starts Ollama CPU-only
REM (OLLAMA_NUM_GPU=0) so the video pipeline can use the GPU. That setting
REM belongs to whichever project started the server first - a server that
REM is already running keeps the mode it was started with. If you need a
REM specific mode, stop Ollama yourself first; this script deliberately
REM will not stop it out from under the other running projects.

set "OLLAMA_URL=http://localhost:11434"
set "OLLAMA_PARALLEL=2"

echo Checking Ollama...

where ollama >nul 2>&1

if errorlevel 1 (
    echo.
    echo ERROR: Ollama is not installed.
    echo Please run setup.bat first.
    exit /b 1
)

echo Ollama detected.

REM Is a server already answering? Reuse it rather than restarting a
REM shared service out from under the other running projects.

curl -s --max-time 2 "%OLLAMA_URL%/api/tags" >nul 2>&1

if not errorlevel 1 (
    echo Ollama already running - reusing the shared instance.
    echo Its accelerator mode was chosen by whichever project started it.
    goto :ollama_ready
)

echo Starting Ollama on CPU...

REM OLLAMA_NO_CLOUD keeps inference local; OLLAMA_NUM_PARALLEL lets several
REM projects generate at the same time instead of one waiting on the
REM other. Both apply to the server this script starts.

set "OLLAMA_NUM_GPU=0"
set "OLLAMA_VULKAN=0"
set "OLLAMA_NO_CLOUD=1"
set "OLLAMA_NUM_PARALLEL=%OLLAMA_PARALLEL%"

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
REM The debugging port is THIS project's own: two projects sharing one
REM browser would fight over it, because the download redirect the provider
REM applies is a browser-wide DevTools setting.

set "SNAPGENAI_DEBUG_PORT=9226"
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

REM Is a Chrome already listening on this project's port? If so it is
REM this project's own browser from an earlier run - reuse it rather
REM than starting a second one that Chrome would only forward to.

curl -s --max-time 2 "http://%SNAPGENAI_DEBUG_HOST%:%SNAPGENAI_DEBUG_PORT%/json/version" >nul 2>&1

if not errorlevel 1 (
    echo SnapGenAI Chrome already listening on port %SNAPGENAI_DEBUG_PORT% - reusing it.
    goto :after_chrome
)

start "Hear His Voice - SnapGenAI Chrome" "%CHROME_EXE%" --remote-debugging-address=%SNAPGENAI_DEBUG_HOST% --remote-debugging-port=%SNAPGENAI_DEBUG_PORT% --user-data-dir="%SNAPGENAI_CHROME_PROFILE%" --start-maximized "https://snapgen.ai/"

echo SnapGenAI Chrome launched.
echo.
echo Selenium will attach to this browser on port %SNAPGENAI_DEBUG_PORT%.
echo.

timeout /t 3 /nobreak >nul

:after_chrome

REM ---------------------------------------------------------------------------
REM Optional: auto-start a Cloudflare quick tunnel so Instagram uploads
REM work out of the box. Instagram's servers fetch the video from a
REM public URL, so publishing must be done through a publicly reachable
REM address. If cloudflared is not installed, Hear His Voice still starts
REM normally in local-only mode (YouTube uploads are unaffected).

REM ---------------------------------------------------------------------------
REM Server port and tunnel
REM
REM Each project gets its own port and therefore its own tunnel URL, so
REM several can run side by side. config/server.json -> port is the single
REM source of truth; APP_PORT passes it to the Python server. Read the port
REM once here so this script and web/server.py can never disagree.

set "SERVER_PORT="

if not "%APP_PORT%"=="" (
    set "SERVER_PORT=%APP_PORT%"
) else (
    for /f "usebackq delims=" %%p in (`python -c "import json,pathlib;print(json.loads(pathlib.Path('config/server.json').read_text(encoding='utf-8')).get('port',''))"`) do set "SERVER_PORT=%%p"
)

if "%SERVER_PORT%"=="" set "SERVER_PORT=8003"

set "APP_PORT=%SERVER_PORT%"

echo Server port: %SERVER_PORT%
echo.

REM Each project needs its own tunnel log, or they overwrite each other's
REM and stop being able to read their own URL.

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

start "hhv-cloudflared" /MIN cmd /c ""%CLOUDFLARED_EXE%" tunnel --url http://localhost:%SERVER_PORT% --no-autoupdate > "%TUNNEL_LOG%" 2>&1"

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

REM ---------------------------------------------------------------------------
REM Open the UI in a browser automatically.
REM
REM The tunnel URL is known before the server starts listening, so this runs
REM in the background and waits for the port to accept a connection before
REM opening anything - otherwise the browser would show a connection error.
REM
REM Set APP_OPEN_BROWSER=0 to skip this (useful over SSH, or when you only
REM want the console).

if not defined APP_OPEN_BROWSER set "APP_OPEN_BROWSER=1"

if "%APP_OPEN_BROWSER%"=="1" (

    if exist "%~dp0scripts\open_in_browser.ps1" (

        start "" /B powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\open_in_browser.ps1" -Port %SERVER_PORT% -Url "%PUBLIC_URL%" -TimeoutSeconds 120

    ) else (

        echo NOTE: scripts\open_in_browser.ps1 not found - skipping auto-open.
        echo Open manually: %PUBLIC_URL%
        echo.

    )

)

python -m web.server

set "EXIT_CODE=%errorlevel%"

REM ---------------------------------------------------------------------------
REM Stop only THIS project's tunnel.
REM
REM Every running project has its own cloudflared process, so killing
REM them by image name would tear down the tunnels belonging to the
REM other projects. scripts\stop_tunnel.ps1 matches on this project's
REM port instead - each tunnel's command line contains the local port
REM it serves. It is a no-op when no tunnel is running.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_tunnel.ps1" -Port %SERVER_PORT%

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