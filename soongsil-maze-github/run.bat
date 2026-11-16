@echo off
setlocal
cd /d "%~dp0"

rem Use Python 3.10+, skipping unavailable or broken launchers.
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
    if not errorlevel 1 (
        py -3 -X utf8 main.py %*
        goto done
    )
)

set "MAZE_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%MAZE_PYTHON%" (
    "%MAZE_PYTHON%" -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
    if not errorlevel 1 (
        "%MAZE_PYTHON%" -X utf8 main.py %*
        goto done
    )
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
    if not errorlevel 1 (
        python -X utf8 main.py %*
        goto done
    )
)

echo Python 3.10 or later was not found. Install Python, then run this file again.
pause
exit /b 1

:done
if errorlevel 1 pause
endlocal

