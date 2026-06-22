# verify_tar.ps1
# Quick smoke test: pack the project using the launcher's helpers,
# then list the resulting tar to verify CJK filenames were preserved.

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

Add-Type -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool SetConsoleCP(uint wCodePageID);
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool SetConsoleOutputCP(uint wCodePageID);
'@ -Name "ConsoleEncoding" -Namespace "HpcLauncher" -ErrorAction SilentlyContinue
[HpcLauncher.ConsoleEncoding]::SetConsoleCP(65001)      | Out-Null
[HpcLauncher.ConsoleEncoding]::SetConsoleOutputCP(65001) | Out-Null

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$tarPath     = Join-Path $env:TEMP ("healthcare_hpc_verify_$(Get-Date -Format 'yyyyMMddHHmmss').tar.gz")
Write-Host "ProjectRoot : $projectRoot"
Write-Host "Tar        : $tarPath"

$parent = Split-Path -Parent $projectRoot
$folder = Split-Path -Leaf $projectRoot

$exclude = @(
    "$folder/outputs", "$folder/outputs_hpc", "$folder/logs_hpc",
    "$folder/env_py311", "$folder/*.pyc", "$folder/__pycache__", "$folder/*.log"
)

# Use Start-Process to correctly pass CJK paths as UTF-8 argv.
$tarArgs = [System.Collections.Generic.List[string]]::new()
$tarArgs.Add("-czf"); $tarArgs.Add($tarPath)
foreach ($ex in $exclude) { $tarArgs.Add("--exclude=$ex") }
$tarArgs.Add("-C"); $tarArgs.Add($parent); $tarArgs.Add($folder)

$tmpOut = Join-Path $env:TEMP ([IO.Path]::GetRandomFileName())
$tmpErr = Join-Path $env:TEMP ([IO.Path]::GetRandomFileName())
$p = Start-Process -FilePath tar.exe -ArgumentList $tarArgs.ToArray() -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr -NoNewWindow -Wait -PassThru
$outText = [string]::Empty; $errText = [string]::Empty
if (Test-Path -LiteralPath $tmpOut) { try { $outText = Get-Content -LiteralPath $tmpOut -Raw -Encoding UTF8 -ErrorAction Stop } catch { }; Remove-Item $tmpOut -Force -ErrorAction SilentlyContinue }
if (Test-Path -LiteralPath $tmpErr) { try { $errText = Get-Content -LiteralPath $tmpErr -Raw -Encoding UTF8 -ErrorAction Stop } catch { }; Remove-Item $tmpErr -Force -ErrorAction SilentlyContinue }
Write-Host "[tar create] exit=$($p.ExitCode)"
if ($outText) { Write-Host $outText }
if ($errText) { Write-Host $errText -ForegroundColor DarkYellow }
if ($p.ExitCode -ne 0) { throw "tar create failed" }

$sizeMB = [math]::Round((Get-Item -LiteralPath $tarPath).Length / 1MB, 2)
Write-Host "[tar create] OK, size=$sizeMB MB"

# Now list the tar contents to verify Chinese folder name is preserved.
$listArgs = @("-tzf", $tarPath)
$tmpOut2 = Join-Path $env:TEMP ([IO.Path]::GetRandomFileName())
$tmpErr2 = Join-Path $env:TEMP ([IO.Path]::GetRandomFileName())
$p2 = Start-Process -FilePath tar.exe -ArgumentList $listArgs -RedirectStandardOutput $tmpOut2 -RedirectStandardError $tmpErr2 -NoNewWindow -Wait -PassThru
$entries = Get-Content -LiteralPath $tmpOut2 -Encoding UTF8
$err2 = if (Test-Path -LiteralPath $tmpErr2) { Get-Content -LiteralPath $tmpErr2 -Raw -Encoding UTF8 -ErrorAction SilentlyContinue } else { [string]::Empty }
Remove-Item $tmpOut2,$tmpErr2 -Force -ErrorAction SilentlyContinue
Write-Host "[tar list] exit=$($p2.ExitCode), total entries=$($entries.Count)"
if ($err2) { Write-Host $err2 -ForegroundColor DarkYellow }

$topLevel = ($entries | ForEach-Object { ($_ -split '/', 2)[0] } | Sort-Object -Unique)
Write-Host "[tar list] top-level dirs: $($topLevel -join ', ')"

$chineseMatch = $entries | Where-Object { $_ -match '医疗|模型|系统|18-' } | Select-Object -First 5
if ($chineseMatch) {
    Write-Host "[tar list] Chinese entries sample:" -ForegroundColor Green
    $chineseMatch | ForEach-Object { Write-Host "  - $_" }
} else {
    Write-Host "[tar list] WARNING: no Chinese entries found" -ForegroundColor Red
}

Remove-Item $tarPath -Force -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "VERIFY DONE" -ForegroundColor Cyan
