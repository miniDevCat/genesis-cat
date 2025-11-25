@echo off
echo ========================================
echo Genesis Real Image Generation
echo ========================================
echo.
echo Starting real image generation interface...
echo.
echo IMPORTANT:
echo - First run will download SD 1.5 model (about 4GB)
echo - Requires: diffusers, transformers, accelerate
echo - GPU recommended for faster generation
echo.
echo ========================================
echo.

e:\Comfyu3.13---test\python_embeded\python.exe gradio_real.py

pause
