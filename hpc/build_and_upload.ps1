# build_and_upload.ps1
# 1) Pack the Healthcare AI project on Windows via Python (UTF-8 safe,
#    avoids CP936/GBK issues with tar.exe when paths contain CJK chars).
# 2) SCP the tarball to the HPC login node.
# 3) Over SSH: extract it, create a Python virtualenv, install deps,
#    then submit SLURM array jobs.
#
# PREREQUISITES (run ONCE before using this script):
#   1. Python 3.9+ installed and on PATH (only for the pack step).
#   2. OpenSSH client (ssh.exe / scp.exe) is installed - included in
#      Windows 10+ and Windows Server 2019+. Verify with:  ssh -V
#   3. Your SSH public key is in ~/.ssh/authorized_keys on the HPC
#      login node.
#   4. You can run the following WITHOUT a password prompt in PowerShell:
#          ssh -i ~/.ssh/scno_key sconqw05b@scno.hpccube.com hostname
#
# TYPICAL USAGE:
#   powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass `
#       -File "e:\各种PY程序\18-医疗AI模型系统\hpc\build_and_upload.ps1"

[CmdletBinding()]
param(
    [string]$ProjectRoot,
    [string]$HpcUser,
    [string]$HpcHost,
    [int]   $HpcPort,
    [string]$HpcKey,
    [string]$HpcProjectDir,
    [string]$RemoteTar,
    [string]$LocalTar
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

# ---------------------------------------------------------------------------
# Defaults are applied in the script body (never inside param()). This keeps
# the file parseable by Windows PowerShell 5.1 regardless of the system
# codepage. The script root is resolved by $PSScriptRoot - a built-in
# variable set by PowerShell to the directory of the currently-running
# script. The project root is always its parent (so we pick up
# 18-医疗AI模型系统/ from inside hpc/).
# ---------------------------------------------------------------------------
if ([string]::IsNullOrEmpty($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
}
if ([string]::IsNullOrEmpty($HpcUser))       { $HpcUser = "sconqw05b" }
if ([string]::IsNullOrEmpty($HpcHost))       { $HpcHost = "scno.hpccube.com" }
if ($HpcPort -eq 0)                          { $HpcPort = 22 }
if ([string]::IsNullOrEmpty($HpcKey))        { $HpcKey = Join-Path $env:USERPROFILE ".ssh\scno_key" }
if ([string]::IsNullOrEmpty($HpcProjectDir)) { $HpcProjectDir = "/public/home/scno/sconqw05b/project" }
if ([string]::IsNullOrEmpty($RemoteTar))     { $RemoteTar = "healthcare_hpc_payload.tar.gz" }
if ([string]::IsNullOrEmpty($LocalTar)) {
    $stamp = Get-Date -Format "yyyyMMddHHmmss"
    $LocalTar = Join-Path $env:TEMP ("healthcare_hpc_" + $stamp + ".tar.gz")
}

# ============================================================
# Helpers
# ============================================================
function Write-Step([string]$Msg) {
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Msg" -ForegroundColor Cyan
}
function Write-Ok([string]$Msg) {
    Write-Host "  OK $Msg" -ForegroundColor Green
}

function Invoke-Native([string]$Cmd, [string[]]$ArgList) {
    # Start a child process, capture its stdout/stderr into temp files,
    # wait for it, then emit them. Uses Start-Process so PowerShell
    # handles the Unicode command line.
    $display = ($ArgList | ForEach-Object {
        if ($_ -match '\s|["'']') { '"{0}"' -f ($_ -replace '"', '\"') } else { $_ }
    }) -join ' '
    Write-Host "  > $Cmd $display"

    $tmpOut = Join-Path $env:TEMP ([IO.Path]::GetRandomFileName())
    $tmpErr = Join-Path $env:TEMP ([IO.Path]::GetRandomFileName())

    $spArgs = @{
        FilePath               = $Cmd
        ArgumentList           = $ArgList
        RedirectStandardOutput = $tmpOut
        RedirectStandardError  = $tmpErr
        NoNewWindow            = $true
        Wait                   = $true
        PassThru               = $true
        ErrorAction            = "Stop"
    }

    try {
        $p = Start-Process @spArgs
        $exitCode = $p.ExitCode
    } finally {
        if (Test-Path -LiteralPath $tmpOut) {
            try   { $text = Get-Content -LiteralPath $tmpOut -Raw -Encoding UTF8 -ErrorAction Stop }
            catch { $text = [string]::Empty }
            if (-not [string]::IsNullOrWhiteSpace($text)) { Write-Host $text }
            Remove-Item -LiteralPath $tmpOut -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path -LiteralPath $tmpErr) {
            try   { $text = Get-Content -LiteralPath $tmpErr -Raw -Encoding UTF8 -ErrorAction Stop }
            catch { $text = [string]::Empty }
            if (-not [string]::IsNullOrWhiteSpace($text)) { Write-Host $text -ForegroundColor DarkYellow }
            Remove-Item -LiteralPath $tmpErr -Force -ErrorAction SilentlyContinue
        }
    }

    if ($null -eq $exitCode -or $exitCode -ne 0) {
        throw "Command failed (ExitCode=$exitCode): $Cmd $display"
    }
}

# ============================================================
# Step 0 - Validate required files / SSH key
# ============================================================
Write-Step "0/5 CONFIG"
Write-Host "  ProjectRoot    : $ProjectRoot"
Write-Host "  HPC login      : ${HpcUser}@${HpcHost}:${HpcPort}"
Write-Host "  SSH private key: $HpcKey"
Write-Host "  HPC project dir: $HpcProjectDir"
Write-Host "  Local tarball  : $LocalTar"

$requiredFiles = @(
    (Join-Path $ProjectRoot "requirements.txt"),
    (Join-Path $ProjectRoot "hpc\jobs\token_valuation.slurm"),
    (Join-Path $ProjectRoot "hpc\hpc_valuation_worker.py"),
    (Join-Path $ProjectRoot "hpc\pack_tar_python.py")
)
foreach ($f in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $f)) { throw "Missing required file: $f" }
}
if (-not (Test-Path -LiteralPath $HpcKey)) {
    throw "SSH private key missing: $HpcKey . Put your private key there, then re-run."
}
Write-Ok "Pre-flight checks passed"

# ============================================================
# Step 1 - Informational local Python check
# ============================================================
Write-Step "1/5 LOCAL PYTHON CHECK"
try {
    $out = & python -c "import os,sys; print(sys.version.split()[0])" 2>&1
    Write-Ok "Python $out available"
} catch {
    throw "Python interpreter not detected on PATH. Python is required to build the CJK-safe tarball."
}

# ============================================================
# Step 2 - Pack via Python (UTF-8 safe, avoids tar.exe CP936)
# ============================================================
Write-Step "2/5 PACK PROJECT (Python tar.gz, UTF-8)"

# Packing is delegated to a small Python script (written to TEMP, then
# removed). Python's tarfile module uses UTF-8 natively, so paths with
# CJK characters survive the round-trip untouched, unlike tar.exe on
# Chinese Windows where the console CP may be locked to 936/GBK.
$packer = Join-Path $env:TEMP ("hpc_pack_" + (Get-Date -Format "yyyyMMddHHmmss") + ".py")
$packerContent = @'
import os, sys, tarfile

project_root = sys.argv[1]
target_tar   = sys.argv[2]

EXCLUDE_DIRS = {"outputs", "outputs_hpc", "logs_hpc", "env_py311", "__pycache__"}
EXCLUDE_NAMES = {"__pycache__"}
EXCLUDE_EXTS = {".pyc", ".log"}

def should_exclude(path):
    rel = os.path.relpath(path, project_root).replace("\\", "/")
    parts = rel.split("/")
    if any(p in EXCLUDE_DIRS for p in parts):
        return True
    name = parts[-1]
    if name in EXCLUDE_NAMES: return True
    ext = os.path.splitext(name)[1].lower()
    if ext in EXCLUDE_EXTS: return True
    return False

project_folder = os.path.basename(project_root)

file_count = 0
dir_count  = 0
with tarfile.open(target_tar, "w:gz", encoding="utf-8") as tar:
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]
        for fname in files:
            fpath = os.path.join(root, fname)
            if should_exclude(fpath): continue
            rel = os.path.relpath(fpath, project_root).replace("\\", "/")
            tar.add(fpath, arcname = (project_folder + "/" + rel), recursive = False)
            file_count += 1
        for dname in dirs:
            dpath = os.path.join(root, dname)
            rel = os.path.relpath(dpath, project_root).replace("\\", "/")
            tar.add(dpath, arcname = (project_folder + "/" + rel + "/"), recursive = False)
            dir_count += 1

size_mb = round(os.path.getsize(target_tar) / 1024 / 1024, 3)
print(f"Packed {file_count} files + {dir_count} dirs -> {target_tar} ({size_mb} MB)")
'@
Set-Content -LiteralPath $packer -Value $packerContent -Encoding UTF8 -NoNewline

try {
    $pythonArgs = @($packer, $ProjectRoot, $LocalTar)
    Invoke-Native -Cmd "python.exe" -ArgList $pythonArgs
} finally {
    Remove-Item -LiteralPath $packer -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath $LocalTar)) {
    throw "Pack script ran but $LocalTar was not created."
}
Write-Ok "Packed: $LocalTar ($([math]::Round((Get-Item -LiteralPath $LocalTar).Length/1MB,2)) MB)"

# ============================================================
# Step 3 - SSH connectivity check
# ============================================================
Write-Step "3/5 SSH CONNECTIVITY CHECK"

$sshBase = [System.Collections.Generic.List[string]]::new()
$sshBase.Add("-i");  $sshBase.Add($HpcKey)
$sshBase.Add("-p");  $sshBase.Add([string]$HpcPort)
$sshBase.Add("-o");  $sshBase.Add("StrictHostKeyChecking=no")
$sshBase.Add("-o");  $sshBase.Add("UserKnownHostsFile=NUL")
$sshBase.Add("-o");  $sshBase.Add("LogLevel=ERROR")
$sshBase.Add("-o");  $sshBase.Add("IdentitiesOnly=yes")

$sshFull = @()
foreach ($x in $sshBase) { $sshFull += $x }
$sshFull += @("${HpcUser}@${HpcHost}", "hostname")

try {
    Invoke-Native -Cmd "ssh.exe" -ArgList $sshFull
    Write-Ok "SSH to HPC login node works"
} catch {
    throw "SSH connection failed. Check VPN / IP whitelist / private key, then re-run."
}

# ============================================================
# Step 4 - Upload tarball + extract on HPC side
# ============================================================
Write-Step "4/5 UPLOAD + EXTRACT ON HPC"

$ensureDirCmd = "mkdir -p `"$HpcProjectDir/outputs_hpc`" `"$HpcProjectDir/logs_hpc`" `"$HpcProjectDir`" ; echo OK"
$ensureDirArg = @()
foreach ($x in $sshBase) { $ensureDirArg += $x }
$ensureDirArg += @("${HpcUser}@${HpcHost}", $ensureDirCmd)
Invoke-Native -Cmd "ssh.exe" -ArgList $ensureDirArg

$remoteTarget = "${HpcUser}@${HpcHost}:${HpcProjectDir}/${RemoteTar}"

$scpArg = @()
foreach ($x in $sshBase) { $scpArg += $x }
$scpArg += @($LocalTar, $remoteTarget)
Write-Step "   4.1 scp upload tarball"
Invoke-Native -Cmd "scp.exe" -ArgList $scpArg
Write-Ok "Uploaded to $remoteTarget"

Write-Step "   4.2 extract tarball on HPC"
$extractCmd = "set -e ; cd `"$HpcProjectDir`" ; tar -xzf `"$RemoteTar`" --strip-components=1 ; rm -f `"$RemoteTar`" ; echo '--- ls top ---' ; ls -1 | head -n 20"
$extractArg = @()
foreach ($x in $sshBase) { $extractArg += $x }
$extractArg += @("${HpcUser}@${HpcHost}", $extractCmd)
Invoke-Native -Cmd "ssh.exe" -ArgList $extractArg
Write-Ok "Extracted"

Remove-Item -LiteralPath $LocalTar -Force -ErrorAction SilentlyContinue

# ============================================================
# Step 5 - HPC side: create virtualenv + install deps + submit SLURM
# ============================================================
Write-Step "5/5 HPC SIDE - CREATE VENV + INSTALL DEPS + SUBMIT SLURM"

# 2026-06 FIX:
#   - sbatch 前必须先 cd 到 $HpcProjectDir（否则 slurm 脚本相对路径可能解析失败）
#   - sbatch 失败立刻退出 set -e；JOBID 为空视为失败
#   - 额外打印 hostname / squeue 给用户自检
#   - 注意：PowerShell @"..."@ 会展开 $() 子表达式和 $variable，
#     因此所有由远端 bash 执行的 $() 必须用 `$` 反引号转义。
$setupAndSubmit = @"
set -e
cd "$HpcProjectDir"
echo "== HPC node info =="
hostname
uname -a
echo "== Create/activate virtualenv =="
if [ ! -d env_py311 ]; then
    python3 -m venv env_py311
fi
source env_py311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || pip install -r requirements.txt
python -c "import pandas,numpy,sklearn,yaml; print('python env OK')"
deactivate
echo "== Dataset check =="
if [ -f healthcare_token_A_B_100M.csv ]; then
    echo "Dataset found: healthcare_token_A_B_100M.csv (`$(du -m healthcare_token_A_B_100M.csv | cut -f1) MB)"
else
    echo "WARN: dataset healthcare_token_A_B_100M.csv not found in $HpcProjectDir"
    echo "  Worker will fall back to synthetic data for benchmarking only."
fi
echo "== Submit SLURM =="
cd "$HpcProjectDir"
JOBID=`$(sbatch --parsable hpc/jobs/token_valuation.slurm)
if [ -z "`$JOBID" ]; then
    echo "ERROR: sbatch returned empty JOBID. Abort."
    exit 1
fi
echo "SLURM_JOB_ID=`$JOBID"
echo "Monitor queue : squeue -u $HpcUser"
echo "Check outputs : ls -lh $HpcProjectDir/outputs_hpc/"
echo "Check logs   : ls -lh $HpcProjectDir/logs_hpc/"
echo "First 30 lines of stderr (best-effort, may not exist yet):"
( tail -n 30 $HpcProjectDir/logs_hpc/slurm_`$JOBID`_*.err 2>/dev/null ) || true
echo "== squeue at submit time =="
squeue -u $HpcUser || true
"@

$submitArg = @()
foreach ($x in $sshBase) { $submitArg += $x }
$submitArg += @("${HpcUser}@${HpcHost}", $setupAndSubmit)
Invoke-Native -Cmd "ssh.exe" -ArgList $submitArg

# ============================================================
# Done
# ============================================================
Write-Step "DONE"
Write-Host "  Monitor on HPC login node:"
Write-Host "    ssh ${HpcUser}@${HpcHost} 'squeue -u ${HpcUser}'"
Write-Host "    ssh ${HpcUser}@${HpcHost} 'ls -lh ${HpcProjectDir}/outputs_hpc/'"
Write-Host "  After all array jobs finish, generate aggregate report:"
Write-Host "    cd ${HpcProjectDir} && source env_py311/bin/activate && python hpc/aggregate_hpc_reports.py --dir outputs_hpc"
Write-Host ""
