# migrate-claude.ps1
# Move Claude Code config + VSCode extensions to drive D:.
#
# REQUIRED before running: fully quit VSCode (including all claude sessions).
# The script self-checks and refuses to run if VSCode/claude is still running.
#
# How to run (in PowerShell):
#   powershell -NoProfile -ExecutionPolicy Bypass -File G:\deepsearch-agents-main\migrate-claude.ps1

$ErrorActionPreference = "Stop"

# ---------- 0. Safety check: VSCode / claude must be exited ----------
$running = Get-Process -Name "Code","claude","claude-code" -ErrorAction SilentlyContinue
if ($running) {
    Write-Host ""
    Write-Host "[!] Still-running processes detected. Fully quit VSCode and claude first, then re-run:" -ForegroundColor Red
    $running | Select-Object Name, Id, Path | Format-Table -AutoSize
    exit 1
}

# ---------- Paths ----------
$configSrc = Join-Path $env:USERPROFILE ".claude"
$jsonSrc   = Join-Path $env:USERPROFILE ".claude.json"
$extSrc    = Join-Path $env:USERPROFILE ".vscode\extensions"

$configDst = "D:\ClaudeConfig"
$extDst    = "D:\VSCodeExtensions"

function Test-RoboOk {
    # robocopy exit code < 8 means success (1 = files copied, 0 = nothing to do)
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed, exit code $LASTEXITCODE" }
}

# ---------- 1. Move config dir contents -> D:\ClaudeConfig ----------
Write-Host "[1/5] Moving ~/.claude -> $configDst ..." -ForegroundColor Cyan
robocopy $configSrc $configDst /E /MOVE /NFL /NDL /NJH /NJS /R:2 /W:2 | Out-Null
Test-RoboOk

# ---------- 2. Move ~/.claude.json -> D:\ClaudeConfig\.claude.json ----------
Write-Host "[2/5] Moving ~/.claude.json -> $configDst\.claude.json ..." -ForegroundColor Cyan
if (Test-Path $jsonSrc) {
    Move-Item $jsonSrc (Join-Path $configDst ".claude.json") -Force
} else {
    Write-Host "    (~/.claude.json not found, skipped)" -ForegroundColor Yellow
}

# ---------- 3. Move VSCode extensions -> D:\VSCodeExtensions ----------
Write-Host "[3/5] Moving VSCode extensions -> $extDst ..." -ForegroundColor Cyan
robocopy $extSrc $extDst /E /MOVE /NFL /NDL /NJH /NJS /R:2 /W:2 | Out-Null
Test-RoboOk

# Clean up any empty source dirs robocopy may leave behind
if (Test-Path $extSrc)    { Remove-Item $extSrc    -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path $configSrc) { Remove-Item $configSrc -Recurse -Force -ErrorAction SilentlyContinue }

# ---------- 4. Create junctions at the old paths pointing to D: ----------
Write-Host "[4/5] Creating junctions ..." -ForegroundColor Cyan
# 4a. extensions: required, so VSCode still finds its extensions at the old path
New-Item -ItemType Junction -Path $extSrc -Target $extDst | Out-Null
# 4b. .claude: fallback, so the old path still resolves if some client ignores the env var
New-Item -ItemType Junction -Path $configSrc -Target $configDst | Out-Null

# ---------- 5. Set user-level env var CLAUDE_CONFIG_DIR ----------
Write-Host "[5/5] Setting env var CLAUDE_CONFIG_DIR = $configDst ..." -ForegroundColor Cyan
[Environment]::SetEnvironmentVariable("CLAUDE_CONFIG_DIR", $configDst, "User")

# ---------- Summary ----------
Write-Host ""
Write-Host "===== Migration done =====" -ForegroundColor Green
Write-Host "  config dir   : $configDst"
Write-Host "  .claude.json : $configDst\.claude.json"
Write-Host "  VSCode ext   : $extDst"
Write-Host "  env var      : CLAUDE_CONFIG_DIR = $configDst (User)"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Reboot (or log off and back on) so the env var applies to new processes."
Write-Host "  2. Reopen VSCode, confirm Claude Code works and figma MCP is still present (/mcp)."
Write-Host "  3. Delete this script once everything is verified (optional)."
