@echo off
setlocal
cd /d "%~dp0"
set "MAZE_GUI_PYTHON="

where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys, tkinter; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)"') do set "MAZE_GUI_PYTHON=%%P"
        goto launch
    )
)

set "MAZE_GUI_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%MAZE_GUI_PYTHON%" (
    "%MAZE_GUI_PYTHON%" -c "import sys, tkinter; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
    if not errorlevel 1 goto launch
)

set "MAZE_GUI_PYTHON="
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys, tkinter; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)"') do set "MAZE_GUI_PYTHON=%%P"
        goto launch
    )
)
echo Python 3.10+ with Tkinter is required. Install Python and try again.
pause
exit /b 1

:launch
for %%P in ("%MAZE_GUI_PYTHON%") do set "MAZE_GUI_PYTHONW=%%~dpPpythonw.exe"
if exist "%MAZE_GUI_PYTHONW%" (
    start "" "%MAZE_GUI_PYTHONW%" -X utf8 "%~dp0gui.py" %*
    exit /b 0
)
"%MAZE_GUI_PYTHON%" -X utf8 "%~dp0gui.py" %*
if errorlevel 1 pause
