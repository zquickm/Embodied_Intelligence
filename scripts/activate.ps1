# ============================================================
#  Isaac Sim / Isaac Lab 环境启动脚本
#
#  用法（注意前面有 ". " 点号和一个空格）：
#      . D:\work\embodied-intelligence\scripts\activate.ps1
#      或在仓库根目录： . .\scripts\activate.ps1
#
#  注意：本文件必须保存为 UTF-8 with BOM，
#        否则 Windows PowerShell 5.1 会按 GBK 读取导致中文乱码、引号解析错误。
# ============================================================

if ($MyInvocation.InvocationName -ne '.') {
    Write-Warning "请用点号方式运行：. <脚本路径>"
    Write-Warning "否则设置的环境变量不会在当前窗口生效。"
}

# ---- 1. 接受 Omniverse EULA（非交互，否则 import isaacsim 会卡住等待输入）
$env:OMNI_KIT_ACCEPT_EULA = "YES"

# ---- 2. 临时目录指向 D 盘（C 盘只剩几 GB，pip 下载大包会爆）
New-Item -ItemType Directory -Force -Path D:\isaac\temp | Out-Null
$env:TEMP = "D:\isaac\temp"
$env:TMP  = "D:\isaac\temp"

# ---- 3. pip 缓存目录
New-Item -ItemType Directory -Force -Path D:\isaac\pipcache | Out-Null
$env:PIP_CACHE_DIR = "D:\isaac\pipcache"

# ---- 4. UTF-8 输出
# 中文 Windows 的 GBK 代码页无法编码 Isaac Lab 启动横幅的制表符，
# 不设这个会在训练刚开始时抛 UnicodeEncodeError 直接退出。
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

# ---- 5. 激活 conda 环境 D:\isaac\env
$hook = "F:\Anaconda\shell\condabin\conda-hook.ps1"
if (Test-Path $hook) { . $hook }

$activated = $false
if (Get-Command conda -ErrorAction SilentlyContinue) {
    conda activate D:\isaac\env 2>&1 | Out-Null
    if ($env:CONDA_PREFIX -eq "D:\isaac\env") { $activated = $true }
}

if (-not $activated) {
    Write-Warning "conda 激活失败，改用 PATH 注入"
    $env:CONDA_PREFIX = "D:\isaac\env"
    $env:PATH = "D:\isaac\env;D:\isaac\env\Scripts;D:\isaac\env\Library\bin;$env:PATH"
}

# ---- 6. Isaac 的 Scripts 加入 PATH，可直接敲 isaaclab
$isaacScripts = "D:\isaac\env\Scripts"
if ((Test-Path $isaacScripts) -and ($env:PATH -notlike "*$isaacScripts*")) {
    $env:PATH = "$isaacScripts;$env:PATH"
}

# ---- 7. 状态回显，确认环境真的对
Write-Host ""
Write-Host "================ Isaac env ready ================" -ForegroundColor Green
$py    = & python -c "import sys; print(sys.executable)" 2>&1
$ver   = & python -c "import sys; print(sys.version.split()[0])" 2>&1
$torch = & python -c "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available())" 2>&1
Write-Host ("  python  : {0}" -f $py)
Write-Host ("  version : {0}" -f $ver)
Write-Host ("  torch   : {0}" -f $torch)
Write-Host ("  TEMP    : {0}" -f $env:TEMP)
Write-Host ("  repo    : D:\work\embodied-intelligence")
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  try: isaaclab list_envs --keyword Cartpole"
Write-Host ""