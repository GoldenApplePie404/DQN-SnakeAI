@echo off
setlocal enabledelayedexpansion

:MENU
cls
echo =================================================================
echo.
echo     ###    ####     ######  ##    ##    ###    ##    ## ######## 
echo    ## ##    ##     ##    ## ###   ##   ## ##   ##   ##  ##       
echo   ##   ##   ##     ##       ####  ##  ##   ##  ##  ##   ##       
echo  ##     ##  ##      ######  ## ## ## ##     ## #####    ######   
echo  #########  ##           ## ##  #### ######### ##  ##   ##       
echo  ##     ##  ##     ##    ## ##   ### ##     ## ##   ##  ##       
echo  ##     ## ####     ######  ##    ## ##     ## ##    ## ######## 
echo.
echo =================================================================
echo 请选择要执行的操作：
echo.
echo [1] 检测环境
echo [2] 训练模型
echo [3] 测试模型
echo [4] 可视化模型结构
echo [5] TFLite转C数组
echo [6] Keras模型转TFLite模型
echo [7] 启动TensorBoard 
echo [8] 运行配置UI
echo [9] 设备监控(施工中...)
echo [0] 退出
echo ======================================
set /p choice="请输入选项: "

:: 检查输入是否有效
if "%choice%"=="1" goto CHECK_PYTHON
if "%choice%"=="2" goto TRAIN
if "%choice%"=="3" goto TEST
if "%choice%"=="4" goto VISMODEL
if "%choice%"=="5" goto TFLITE2C
if "%choice%"=="6" goto K2TFLITE
if "%choice%"=="7" goto TENSORBOARD
if "%choice%"=="8" goto CONFIG_UI
if "%choice%"=="9" goto DEVICE_MONITOR
if "%choice%"=="0" exit /b

:: 输入无效时提示
echo 无效输入，请按任意键重新选择...
pause >nul
goto MENU

:CHECK_PYTHON
cls
echo 正在检测 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 未检测到 Python 环境！
    echo 请先安装 Python（推荐 3.11+）并添加到 PATH。
    echo.
    echo 安装方式：
    echo 1. 访问官网下载：https://www.python.org/downloads/
    echo 2. 安装时勾选 "Add Python to PATH"。
    echo.
    pause
    goto MENU
)

:: 显示 Python 版本
for /f "delims=" %%i in ('python --version 2^>^&1') do set "py_version=%%i"
echo 检测到 Python: %py_version%
echo.

:: 运行第三方库检测脚本
echo 正在检测第三方库依赖（执行 r_installer.py）...
python src\tools\r_installer.py
if %errorlevel% neq 0 (
    echo.
    echo 警告：r_installer.py 执行失败！
    echo 可能原因：
    echo - 脚本不存在或路径错误
    echo - 依赖库未安装（如 pip, requests 等）
    echo.
)
pause
goto MENU

:TRAIN
echo 正在启动训练...
python src\trainer\trainer.py
pause
goto MENU

:TEST
echo 正在启动测试...
python src\tools\tester.py
pause
goto MENU

:VISMODEL
echo 正在可视化模型结构...
python src\tools\vismodel.py
pause
goto MENU

:TFLITE2C
echo 正在转换 TFLite 模型 到 C数组...
python src\tools\tflite2c.py
pause
goto MENU

:TENSORBOARD
echo 正在启动 TensorBoard...
start "TensorBoard" cmd /k "tensorboard --logdir logs\tensorboard"
echo TensorBoard 已在新窗口启动。
pause
goto MENU

:CONFIG_UI
echo 正在启动配置UI...
python src\tools\config_ui.py
echo 配置UI已关闭，返回主菜单。
pause
goto MENU

:K2TFLITE
echo 正在转换 Keras 到 TFLite...
python src\tools\k2tflite.py
pause
goto MENU

:DEVICE_MONITOR
echo 正在启动设备监控...
python src\tools\device_monitor.py
pause
goto MENU
