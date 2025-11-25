@echo off
chcp 65001 >nul
title Genesis WanVideo - 启动前后端服务

echo ============================================================
echo Genesis WanVideo 视频生成系统
echo ============================================================
echo.

:: 设置颜色
color 0A

:: 检查 Python 环境
echo [1/4] 检查 Python 环境...
if not exist "python_embeded\python.exe" (
    echo [错误] 未找到 python_embeded\python.exe
    echo 请确保 Python 嵌入式环境已正确安装
    pause
    exit /b 1
)
echo [✓] Python 环境检查通过
echo.

:: 检查后端文件
echo [2/4] 检查后端文件...
if not exist "genesis\api_wanvideo_server.py" (
    echo [错误] 未找到后端服务文件
    pause
    exit /b 1
)
echo [✓] 后端文件检查通过
echo.

:: 检查前端文件
echo [3/4] 检查前端文件...
if not exist "genesis-web-ui\package.json" (
    echo [错误] 未找到前端项目文件
    pause
    exit /b 1
)
echo [✓] 前端文件检查通过
echo.

:: 启动服务
echo [4/4] 启动服务...
echo.
echo ============================================================
echo 正在启动后端服务 (Flask - 端口 5000)...
echo ============================================================
start "Genesis Backend" cmd /k "python_embeded\python.exe genesis\api_wanvideo_server.py"

:: 等待后端启动
echo 等待后端服务启动 (5秒)...
timeout /t 5 /nobreak >nul

echo.
echo ============================================================
echo 正在启动前端服务 (React - 端口 8000)...
echo ============================================================
start "Genesis Frontend" cmd /k "cd genesis-web-ui && npm run dev"

echo.
echo ============================================================
echo 服务启动完成！
echo ============================================================
echo.
echo 后端地址: http://localhost:5000
echo 前端地址: http://localhost:8000
echo.
echo 提示：
echo - 两个命令行窗口将会打开，请不要关闭它们
echo - 前端启动需要一些时间，请耐心等待
echo - 浏览器会自动打开前端页面
echo - 按 Ctrl+C 可以停止各个服务
echo.
echo ============================================================
echo 按任意键关闭此窗口...
pause >nul
