@echo off
:: 设置编码为 UTF-8，防止中文显示乱码
chcp 65001 >nul

:: 切换到当前批处理文件所在的目录下
cd /d "%~dp0"

:: 定义您的 PowerShell 脚本文件名，请确保与实际文件名一致
set "PS_SCRIPT=init_env.ps1"

:: 检查脚本是否存在
if not exist "%PS_SCRIPT%" (
    echo [错误] 未在当前目录下找到脚本文件: %PS_SCRIPT%
    echo 请确保 %PS_SCRIPT% 与此 .bat 文件放在同一个文件夹内。
    echo.
    pause
    exit /b 1
)

echo 正在启动 PowerShell 脚本

:: 调用 PowerShell 执行脚本
:: -NoProfile: 不加载用户配置文件，加快启动速度
:: -ExecutionPolicy Bypass: 绕过脚本执行策略限制，允许运行未签名的本地脚本
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%"

echo 脚本执行完毕。
pause
