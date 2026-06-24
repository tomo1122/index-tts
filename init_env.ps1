# 提前设置 HuggingFace 国内镜像源环境变量
$env:HF_ENDPOINT = "https://hf-mirror.com"

Write-Host "步骤 1：硬件探测与拦截" -ForegroundColor Cyan
$gpus = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name
$gpuInfo = ""
foreach ($gpu in $gpus) { $gpuInfo += $gpu + " " }
Write-Host "检测到的显卡设备: $gpuInfo" -ForegroundColor Green

$syncParams = @()
if ($gpuInfo -match "NVIDIA") {
    Write-Host "=> 识别为 NVIDIA 架构，应用 CUDA 12.8 配置" -ForegroundColor Green
    # 避开 Windows 下 DeepSpeed 的编译错误，安全地安装 WebUI 推荐配置。
    $syncParams = @("--extra", "webui")
    Write-Host "=> Windows 下不安装 deepspeed 加速" -ForegroundColor Yellow
} elseif ($gpuInfo -match "AMD" -or $gpuInfo -match "Radeon") {
    Write-Host "=> 识别为 AMD 架构。Windows 上 PyTorch 官方不支持 ROCm 加速。请通过 WSL 使用 Linux 版本的安装脚本运行。" -ForegroundColor Red
    Exit
} else {
    Write-Host "=> 未检测到常用显卡" -ForegroundColor Red
    Exit
}

Write-Host ""
Write-Host "步骤 2：环境基础建设 (Git & uv)" -ForegroundColor Cyan
$GIT_REPO = "https://github.com/tomo1122/index-tts.git"
$DIR_NAME = "index-tts"

# 1. 确保在项目目录中
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "当前目录无 pyproject.toml，准备克隆仓库..." -ForegroundColor Yellow

    # 检查 Git 是否安装
    if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
        Write-Host "当前未安装 Git，正在从国内源拉取并自动安装..." -ForegroundColor Yellow
        $gitUrl = "https://mirrors.huaweicloud.com/git-for-windows-local/v2.53.0.windows.1/Git-2.53.0-64-bit.exe"
        & curl.exe -L -o "git-setup.exe" $gitUrl

        Write-Host "正在静默安装 Git (包含 Git LFS)..." -ForegroundColor Yellow
        Start-Process -FilePath "git-setup.exe" -ArgumentList "/SILENT /NORESTART" -Wait
        Remove-Item "git-setup.exe"

        # 刷新当前会话的环境变量
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    }

    Write-Host "开始克隆仓库..." -ForegroundColor Yellow
    git clone $GIT_REPO
    if (-not (Test-Path $DIR_NAME)) {
        Write-Error "克隆失败，请检查仓库链接或网络。"
        Exit
    }
    Set-Location $DIR_NAME
}

# 2. 安装 uv
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "正在安装 uv 用于创建环境..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # 临时将 uv 添加到当前会话的环境变量
    $env:Path += ";$Home\.local\bin"
}

# 3. 拦截并覆写传递依赖
$tomlPath = "pyproject.toml"
if (Test-Path $tomlPath) {
    Write-Host "正在检查并注入依赖覆盖规则..." -ForegroundColor Yellow
    $content = Get-Content $tomlPath -Raw

    # 检查是否已注入过，避免重复写入
    if ($content -notmatch 'override-dependencies\s*=') {
        $targetRule = "`r`noverride-dependencies = [`"onnxruntime ; sys_platform == 'never'`"]"

        # 兼容 Windows (\r\n) 和 Unix (\n) 换行符，安全地在 [tool.uv] 声明行正下方插入配置
        if ($content -match "\[tool\.uv\]\r\n") {
            $content = $content -replace "\[tool\.uv\]\r\n", "[tool.uv]$targetRule`r`n"
        } elseif ($content -match "\[tool\.uv\]\n") {
            $targetRuleUnix = "`noverride-dependencies = [`"onnxruntime ; sys_platform == 'never'`"]"
            $content = $content -replace "\[tool\.uv\]\n", "[tool.uv]$targetRuleUnix`n"
        } else {
            $content = $content -replace "\[tool\.uv\]", "[tool.uv]$targetRule"
        }

        Set-Content -Path $tomlPath -Value $content -Encoding UTF8
        Write-Host "成功向 pyproject.toml 的 [tool.uv] 区块注入 override-dependencies 拦截规则。" -ForegroundColor Green
    } else {
        Write-Host "检测到依赖覆盖规则已存在，跳过注入。" -ForegroundColor Green
    }
}

# 4. 注入 NVIDIA GPU 专用的依赖约束
Write-Host "正在配置 GPU 版本 ONNX Runtime 及辅助依赖..." -ForegroundColor Yellow
uv add "onnxruntime-gpu<1.20" "opencc-python-reimplemented>=0.1.7" "pypinyin-g2pw==0.4.0"

# 5. 同步环境
Write-Host "开始同步环境依赖（使用清华源加速）..." -ForegroundColor Yellow
uv sync $syncParams --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"


Write-Host ""
Write-Host "步骤 3：多音字及 BERT 基础模型下载" -ForegroundColor Cyan
$modelsDir = "indextts_batch/models"
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
}

# 3.1 下载普通话修正版 G2PWModel
$g2pwPath = Join-Path $modelsDir "G2PWModel"
if (-not (Test-Path $g2pwPath)) {
    Write-Host "正在从国内镜像站下载 G2PWModel (大陆版)..." -ForegroundColor Yellow
    $g2pwZip = Join-Path $modelsDir "G2PWModel.zip"
    $g2pwUrl = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
    & curl.exe -L -o $g2pwZip $g2pwUrl

    if (Test-Path $g2pwZip) {
        Write-Host "正在解压模型文件..." -ForegroundColor Yellow
        Expand-Archive -Path $g2pwZip -DestinationPath $modelsDir -Force
        Remove-Item $g2pwZip

        # 兼容性处理：重命名 g2pW.onnx 为 g2pw.onnx (使用安全的两步重命名，解决 Linux 跨平台兼容性)
        $oldOnnxPath = Join-Path $g2pwPath "g2pW.onnx"
        if (Test-Path $oldOnnxPath) {
            $tempOnnxPath = Join-Path $g2pwPath "g2p_temp.onnx"
            Rename-Item -Path $oldOnnxPath -NewName "g2p_temp.onnx" -Force
            Rename-Item -Path $tempOnnxPath -NewName "g2pw.onnx" -Force
            Write-Host "重命名 g2pW.onnx -> g2pw.onnx 成功。" -ForegroundColor Green
        }
    } else {
        Write-Error "G2PWModel 压缩包下载失败，中断后续解压流程。"
        Exit
    }
} else {
    Write-Host "G2PWModel 目录已存在，跳过下载。" -ForegroundColor Green
}

# 3.2 下载 BERT 中文基础模型
$bertPath = Join-Path $modelsDir "bert-base-chinese"
if (-not (Test-Path $bertPath)) {
    Write-Host "正在下载 bert-base-chinese 模型..." -ForegroundColor Yellow
    $bertUrl = "https://www.modelscope.cn/tiansz/bert-base-chinese.git"

    # 初始化 Git LFS
    git lfs install

    # 切入 models 目录进行克隆并返回
    $currentDir = Get-Location
    Set-Location $modelsDir
    git clone $bertUrl
    Set-Location $currentDir

    # 清理 .git 文件夹节省空间，并防止嵌套 Git 模块冲突
    $bertGit = Join-Path $bertPath ".git"
    if (Test-Path $bertGit) {
        Remove-Item -Path $bertGit -Recurse -Force
    }
} else {
    Write-Host "检测到 bert-base-chinese 目录已存在，跳过下载。" -ForegroundColor Green
}


Write-Host ""
Write-Host "步骤 4：核心 checkpoints 下载" -ForegroundColor Cyan
$checkpointsPath = "checkpoints"
$gptModelPath = Join-Path $checkpointsPath "gpt.pth"

if (-not (Test-Path $gptModelPath)) {
    Write-Host "检测到 checkpoints 核心权重缺失，准备拉取 IndexTTS-2 核心权重..." -ForegroundColor Yellow
    uv tool install "modelscope"

    # 使用 uvx 执行免环境变量刷新的模型拉取
    uvx modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
} else {
    Write-Host "检测到 IndexTTS-2 核心权重 checkpoints 已存在，跳过下载。" -ForegroundColor Green
}


Write-Host ""
Write-Host "步骤 5：信息打印" -ForegroundColor Cyan
[System.Console]::ForegroundColor = [System.ConsoleColor]::Green
[System.Console]::WriteLine("环境安装完成，运行 uv run -m indextts_batch.main 进行测试")
[System.Console]::WriteLine("需要修改参考音频，放到example文件夹下，并且修改 main.py 中的音频路径")
[System.Console]::ResetColor()
Write-Host ""
