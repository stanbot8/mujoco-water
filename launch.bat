@echo off
taskkill /f /im multiscale_demo.exe >nul 2>&1
timeout /t 1 /nobreak >nul
del /f "%~dp0build\Release\multiscale_demo.exe" >nul 2>&1
"C:\Program Files\CMake\bin\cmake.exe" --build "%~dp0build" --config Release --target multiscale_demo
if %errorlevel% neq 0 exit /b %errorlevel%
start "" "%~dp0build\Release\multiscale_demo.exe"
