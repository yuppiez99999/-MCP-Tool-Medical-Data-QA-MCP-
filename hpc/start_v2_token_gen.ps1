# start_v2_token_gen.ps1
#
# 用法:
#   powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass `
#       -File "e:\各种PY程序\18-医疗AI模型系统\hpc\start_v2_token_gen.ps1 -Scale 5000
#
# 功能:
#   1) 校验本地 SSH 私钥是否存在
#   2) scp 上传 hpc/gen_large.py 和 hpc/run_v2_token_gen.sh
#      到 超算 ~/token_project/
#   3) 在超算 login 节点 执行 bash run_v2_token_gen.sh <scale>
#   4) 打印最终 tar.gz 路径（下载请在 Web 门户下载或手动 scp）

[CmdletBinding()]
param(
    [string]$HpcUser       = "scnonqw05b",
    [string]$HpcHost       = "scno.hpccube.com",
    [int]$HpcPort        = 22,
    [string]$HpcKey        = (Join-Path $env:USERPROFILE ".ssh\scno_key"),
    [string]$RemoteProjectDir = "~/token_project",
    [int]$Scale           = 5000,
    [switch]$ScaleTestOnly = $false
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

# ------------------------------------------------------------------
# 依赖: ssh.exe / scp.exe
# ------------------------------------------------------------------
function Resolve-Native([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { throw "未找到 $Name，请安装 OpenSSH 客户端（Windows 10+ 默认已自带，在“可选功能”里安装）。" }
    return $cmd.Source
}

$SSH_EXE = Resolve-Native "ssh.exe"
$SCP_EXE = Resolve-Native "scp.exe"

# ------------------------------------------------------------------
# 路径解析
# ------------------------------------------------------------------
$SCRIPT_DIR  = Split-Path -Parent $MyInvocation.MyCommand.Path
$LOCAL_PY    = Join-Path $SCRIPT_DIR "gen_large.py"
$LOCAL_SH    = Join-Path $SCRIPT_DIR "run_v2_token_gen.sh"

if (-not (Test-Path -LiteralPath $LOCAL_PY))  { throw "Missing: $LOCAL_PY" }
if (-not (Test-Path -LiteralPath $LOCAL_SH))  { throw "Missing: $LOCAL_SH" }
if (-not (Test-Path -LiteralPath $HpcKey)) { throw "SSH 私钥不存在: $HpcKey`n 请把私钥放到 $HpcKey" }

function Write-Step([string]$Msg) {
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Msg" -ForegroundColor Cyan
}
function Write-Ok([string]$Msg) {
    Write-Host "  OK $Msg" -ForegroundColor Green
}

Write-Step "参数"
Write-Host ("  HPC User     : $HpcUser"
Write-Host ("  HPC Host     : $HpcHost`:$HpcPort"
Write-Host ("  SSH Key      : $HpcKey"
Write-Host ("  项目目录     : $RemoteProjectDir"
Write-Host ("  Scale        : $Scale"
Write-Host ("  仅小测       : $ScaleTestOnly"

$SSH_BASE = @(
    "-i", $HpcKey,
    "-p", [string]$HpcPort,
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=NUL",
    "-o", "LogLevel=ERROR",
    "-o", "IdentitiesOnly=yes",
    "-o", "ConnectTimeout=15"
)

# ------------------------------------------------------------------
# Step 1 - SSH 连通性
# ------------------------------------------------------------------
Write-Step "[1/3] 测试 SSH 到 ${HpcUser}@${HpcHost}:${HpcPort}"
try {
    $r = & $SSH_EXE @SSH_BASE "${HpcUser}@${HpcHost}" "echo ping_ok"
    if ($LASTEXITCODE -ne 0) { throw "ssh 失败 (ExitCode=$LASTEXITCODE)" }
} catch {
    Write-Warning "SSH 到超算登录节点失败。"
    Write-Warning "  原因可能是：1) Windows 无法解析超算内网 DNS (${HpcHost}); 2) 私钥未在超算 authorized_keys；3) 需先接 VPN。"
    Write-Warning ""
    Write-Warning "请切换到 Web 门户 (https://www.scnet.cn) 并执行："
    Write-Warning ""
    Write-Warning "  mkdir -p ~/token_project/output ~/token_project/logs"
    Write-Warning "  # 然后在你上传到 Web 终端里手动粘贴 hpc/gen_large.py 和 hpc/run_v2_token_gen.sh 的内容"
    Write-Warning "  # 或在本地编辑后复制内容粘贴到 https://www.scnet.cn 的 Web 终端，保存到 ~/token_project/ 后执行："
    Write-Warning "  bash ~/token_project/run_v2_token_gen.sh $Scale"
    exit 1
}
Write-Ok "SSH 可达"

# ------------------------------------------------------------------
# Step 2 - SCP 上传 gen_large.py + run_v2_token_gen.sh
# ------------------------------------------------------------------
Write-Step "[2/3] 上传脚本到超算"
$ensureDirCmd = "mkdir -p `"$RemoteProjectDir/output`" `"$RemoteProjectDir/logs`""
& $SSH_EXE @SSH_BASE "${HpcUser}@${HpcHost}" $ensureDirCmd
if ($LASTEXITCODE -ne 0) { throw "mkdir -p 失败" }

$dest = "${HpcUser}@${HpcHost}:${RemoteProjectDir}/"
& $SCP_EXE @SSH_BASE $LOCAL_PY $LOCAL_SH $dest
if ($LASTEXITCODE -ne 0) { throw "scp 上传失败" }
Write-Ok "已上传 $(Split-Path -Leaf $LOCAL_PY), $(Split-Path -Leaf $LOCAL_SH)"

# ------------------------------------------------------------------
# Step 3 - 在超算端执行 run_v2_token_gen.sh
# ------------------------------------------------------------------
Write-Step "[3/3] 在超算 login 节点执行 run_v2_token_gen.sh ${Scale}"
$remoteCmd = "cd `"$RemoteProjectDir`" && chmod +x run_v2_token_gen.sh && "
if ($ScaleTestOnly) {
    $remoteCmd += "SCALE_TEST_ONLY=1 bash run_v2_token_gen.sh $Scale"
} else {
    $remoteCmd += "bash run_v2_token_gen.sh $Scale"
}
& $SSH_EXE @SSH_BASE -t -t "${HpcUser}@${HpcHost}" $remoteCmd
$exitCode = $LASTEXITCODE

# ------------------------------------------------------------------
# 结束语
# ------------------------------------------------------------------
if ($exitCode -ne 0) {
    Write-Warning "超算执行完成但返回 ExitCode=$exitCode。请在 https://www.scnet.cn 手动检查。"
    exit $exitCode
}
Write-Host ""
Write-Host "=== 操作完成 ===" -ForegroundColor Green
Write-Host "  请登录 https://www.scnet.cn"
Write-Host "  在 Web 终端里执行以下命令来下载 tar.gz:"
Write-Host ""
Write-Host "    ls -lh ~/北数所Token_HPC生成包_*.tar.gz"
Write-Host ""
