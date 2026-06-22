# launch.ps1 - pure ASCII launcher for the HPC deployment script.
#
# Why a separate launcher file?
#   On Chinese Windows, PowerShell 5.1 reads .ps1 files using the
#   system codepage (CP936/GBK) by default. When a script contains
#   literal CJK characters AND also spawns child processes that need
#   to receive UTF-8 arguments (tar.exe, ssh.exe, scp.exe), the two
#   requirements conflict at the parser level.
#
#   The launcher keeps its own source 100% ASCII (so parsing is never
#   ambiguous) and forces UTF-8 process encodings BEFORE it calls the
#   real script. CJK paths are then carried through PowerShell variables
#   (which are always Unicode in memory) rather than through file-content
#   string literals.
#
# Usage:
#   powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass `
#       -File "e:\各种PY程序\18-医疗AI模型系统\hpc\launch.ps1"
#
# Optional forwarded parameters: -HpcUser, -HpcHost, -HpcPort, -HpcKey,
# -HpcProjectDir, -ProjectRoot, -RemoteTar, -LocalTar.

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

# --- Force UTF-8 for child-process I/O. Must happen BEFORE the first
# external-process call in the whole pipeline. ------------------------
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

# SetConsoleCP / SetConsoleOutputCP (Win32) so that child processes
# see UTF-8 input on their stdin / command-line, even when their
# argv comes from non-ASCII PowerShell variables.
Add-Type -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool SetConsoleCP(uint wCodePageID);
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool SetConsoleOutputCP(uint wCodePageID);
'@ -Name "ConsoleEncoding" -Namespace "HpcLauncher" -ErrorAction SilentlyContinue
[HpcLauncher.ConsoleEncoding]::SetConsoleCP(65001)       | Out-Null
[HpcLauncher.ConsoleEncoding]::SetConsoleOutputCP(65001)  | Out-Null

$scriptPath = Join-Path $PSScriptRoot "build_and_upload.ps1"
Write-Host "Launcher -> $scriptPath"

$forwardArgs = @{}
foreach ($k in @('ProjectRoot','HpcUser','HpcHost','HpcPort','HpcKey',
                 'HpcProjectDir','RemoteTar','LocalTar')) {
    $val = $PSBoundParameters[$k]
    if ($val -ne $null -and $val -ne '') {
        if ($k -eq 'HpcPort') { $forwardArgs[$k] = [int]$val }
        else                   { $forwardArgs[$k] = [string]$val }
    }
}

& $scriptPath @forwardArgs
exit $LASTEXITCODE
