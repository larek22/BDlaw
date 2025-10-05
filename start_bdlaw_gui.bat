@echo off
setlocal
cd /d %~dp0

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" start_bdlaw.py --no-compose %*
if errorlevel 1 (
    echo.
    echo Возникла ошибка при запуске BDlaw. Проверьте сообщения выше.
    pause
)
