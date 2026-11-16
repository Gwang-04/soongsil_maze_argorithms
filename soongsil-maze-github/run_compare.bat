@echo off
setlocal
call "%~dp0run.bat" --compare %*
set "MAZE_COMPARE_EXIT=%ERRORLEVEL%"
pause
exit /b %MAZE_COMPARE_EXIT%
