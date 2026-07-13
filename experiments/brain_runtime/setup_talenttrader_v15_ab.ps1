param(
    [string]$Root = "C:\Code\TalentTrader\v15-ab-v4",
    [string]$Model = "gpt-5.6-sol",
    [string]$AIOptimizerHome = "C:\Code\AIOptimizer",
    [string]$BaseCodexHome = (Join-Path $env:USERPROFILE ".codex"),
    [string]$CodexControlRoot = (Join-Path $env:USERPROFILE ".codex\ab-v15-v4")
)

$ErrorActionPreference = "Stop"

function Write-NewFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )

    if (Test-Path -LiteralPath $Path) {
        $existing = [System.IO.File]::ReadAllText($Path)
        if ($existing -ne $Content) {
            throw "Refusing to overwrite existing file with different content: $Path"
        }
        return
    }

    $parent = Split-Path -Parent $Path
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

function New-JunctionIfMissing {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Target
    )

    if (Test-Path -LiteralPath $Path) {
        return
    }
    $parent = Split-Path -Parent $Path
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    New-Item -ItemType Junction -Path $Path -Target $Target | Out-Null
}

function New-HardLinkIfMissing {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Target
    )

    if (Test-Path -LiteralPath $Path) {
        return
    }
    $parent = Split-Path -Parent $Path
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    New-Item -ItemType HardLink -Path $Path -Target $Target | Out-Null
}

function Set-AIOptimizerPluginState {
    param(
        [Parameter(Mandatory = $true)][string]$Config,
        [Parameter(Mandatory = $true)][bool]$Enabled
    )

    $pattern = '(?m)(^\[plugins\."aioptimizer-codex@personal"\]\r?\n)enabled\s*=\s*(?:true|false)'
    $matches = [System.Text.RegularExpressions.Regex]::Matches($Config, $pattern)
    if ($matches.Count -ne 1) {
        throw "Expected exactly one aioptimizer-codex plugin section in the base config."
    }
    $value = if ($Enabled) { "true" } else { "false" }
    return [System.Text.RegularExpressions.Regex]::Replace(
        $Config,
        $pattern,
        "`${1}enabled = $value"
    )
}

function Set-TopLevelTomlString {
    param(
        [Parameter(Mandatory = $true)][string]$Config,
        [Parameter(Mandatory = $true)][string]$Key,
        [Parameter(Mandatory = $true)][string]$Value
    )

    $escapedKey = [System.Text.RegularExpressions.Regex]::Escape($Key)
    $pattern = "(?m)^$escapedKey\s*=.*$"
    $matches = [System.Text.RegularExpressions.Regex]::Matches($Config, $pattern)
    if ($matches.Count -gt 1) {
        throw "Expected at most one top-level $Key setting in the base config."
    }

    $setting = "$Key = `"$Value`""
    if ($matches.Count -eq 1) {
        return [System.Text.RegularExpressions.Regex]::Replace($Config, $pattern, $setting)
    }

    $newline = if ($Config.Contains("`r`n")) { "`r`n" } else { "`n" }
    $firstTable = [System.Text.RegularExpressions.Regex]::Match($Config, '(?m)^\s*\[')
    if ($firstTable.Success) {
        return $Config.Insert($firstTable.Index, "$setting$newline$newline")
    }
    return "$Config$newline$setting$newline"
}

function Set-AutoReviewDefaults {
    param([Parameter(Mandatory = $true)][string]$Config)

    $updated = Set-TopLevelTomlString -Config $Config -Key "approval_policy" -Value "on-request"
    $updated = Set-TopLevelTomlString -Config $updated -Key "approvals_reviewer" -Value "auto_review"
    return Set-TopLevelTomlString -Config $updated -Key "sandbox_mode" -Value "workspace-write"
}

$resolvedRoot = [System.IO.Path]::GetFullPath($Root)
$armA = Join-Path $resolvedRoot "arm-a-plugin-on\workspace"
$armB = Join-Path $resolvedRoot "arm-b-plugin-off\workspace"
[System.IO.Directory]::CreateDirectory($armA) | Out-Null
[System.IO.Directory]::CreateDirectory($armB) | Out-Null

# The model-visible workspace seeds must remain byte-identical across arms.
$seedGitIgnore = ".aioptimizer/`n"
Write-NewFile -Path (Join-Path $armA ".gitignore") -Content $seedGitIgnore
Write-NewFile -Path (Join-Path $armB ".gitignore") -Content $seedGitIgnore

$baseConfigPath = Join-Path $BaseCodexHome "config.toml"
$baseAuthPath = Join-Path $BaseCodexHome "auth.json"
if (-not (Test-Path -LiteralPath $baseConfigPath)) {
    throw "Base Codex config not found: $baseConfigPath"
}
if (-not (Test-Path -LiteralPath $baseAuthPath)) {
    throw "Base Codex auth store not found: $baseAuthPath"
}

$codexHomeA = Join-Path $CodexControlRoot "arm-a"
$codexHomeB = Join-Path $CodexControlRoot "arm-b"
$baseConfig = Set-AutoReviewDefaults -Config ([System.IO.File]::ReadAllText($baseConfigPath))
$configA = Set-AIOptimizerPluginState -Config $baseConfig -Enabled $true
$configB = Set-AIOptimizerPluginState -Config $baseConfig -Enabled $false
Write-NewFile -Path (Join-Path $codexHomeA "config.toml") -Content $configA
Write-NewFile -Path (Join-Path $codexHomeB "config.toml") -Content $configB
New-HardLinkIfMissing -Path (Join-Path $codexHomeA "auth.json") -Target $baseAuthPath
New-HardLinkIfMissing -Path (Join-Path $codexHomeB "auth.json") -Target $baseAuthPath
New-JunctionIfMissing -Path (Join-Path $codexHomeA "plugins") -Target (Join-Path $BaseCodexHome "plugins")
New-JunctionIfMissing -Path (Join-Path $codexHomeB "plugins") -Target (Join-Path $BaseCodexHome "plugins")
if (Test-Path -LiteralPath (Join-Path $BaseCodexHome "skills")) {
    New-JunctionIfMissing -Path (Join-Path $codexHomeA "skills") -Target (Join-Path $BaseCodexHome "skills")
    New-JunctionIfMissing -Path (Join-Path $codexHomeB "skills") -Target (Join-Path $BaseCodexHome "skills")
}

$launcherTemplate = @'
param()

$ErrorActionPreference = "Stop"
$env:CODEX_HOME = "{4}"
$env:AIOPTIMIZER_HOME = "{0}"
$workspace = Join-Path $PSScriptRoot "{1}\workspace"

& codex.cmd `
    -C $workspace `
    --sandbox workspace-write `
    --ask-for-approval on-request `
    --model "{3}"
exit $LASTEXITCODE
'@

$launcherA = $launcherTemplate -f $AIOptimizerHome, "arm-a-plugin-on", "true", $Model, $codexHomeA
$launcherB = $launcherTemplate -f $AIOptimizerHome, "arm-b-plugin-off", "false", $Model, $codexHomeB
Write-NewFile -Path (Join-Path $resolvedRoot "start-arm-a.ps1") -Content $launcherA
Write-NewFile -Path (Join-Path $resolvedRoot "start-arm-b.ps1") -Content $launcherB

$readmeTemplate = @'
# TalentTrader v15 A/B workspaces

These are clean, paired workspaces for the frozen AIOptimizer v15 dogfood A/B.

- `arm-a-plugin-on/workspace`: launch with `start-arm-a.ps1`; AIOptimizer plugin forced on.
- `arm-b-plugin-off/workspace`: launch with `start-arm-b.ps1`; AIOptimizer plugin forced off.
- Both launchers pin model `{0}` and use `workspace-write` with on-request approvals.
- Both frozen configs route eligible approval prompts to Codex auto-review, so the paired runs
  do not require routine human approval while retaining the sandbox boundary.
- Each arm uses a frozen isolated `CODEX_HOME`; the configs differ only in the AIOptimizer enabled bit.
- Both isolated homes hardlink the same protected auth store and junction the same plugin binaries.
- The model-visible seed files are byte-identical. Arm identity exists only in the parent control plane.
- Start a fresh Codex thread for every run. Do not reuse a workspace after a scripted build.
- Create new task/replicate pairs from these seeds rather than resetting completed runs.
- Score only path-anonymized artifact copies, as required by PREREGISTRATION_v15.md.

AIOptimizer source: `{1}`
'@
$readme = $readmeTemplate -f $Model, $AIOptimizerHome
Write-NewFile -Path (Join-Path $resolvedRoot "README.md") -Content $readme

$hashA = (Get-FileHash -LiteralPath (Join-Path $armA ".gitignore") -Algorithm SHA256).Hash
$hashB = (Get-FileHash -LiteralPath (Join-Path $armB ".gitignore") -Algorithm SHA256).Hash
if ($hashA -ne $hashB) {
    throw "Arm seeds are not byte-identical."
}

[PSCustomObject]@{
    root = $resolvedRoot
    arm_a = $armA
    arm_b = $armB
    codex_home_a = $codexHomeA
    codex_home_b = $codexHomeB
    model = $Model
    seed_sha256 = $hashA
    identical = $true
} | ConvertTo-Json -Depth 3
