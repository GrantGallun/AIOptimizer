<#
Run this ONCE, interactively, after (re)locating a Codex control home -- e.g. after
setup_talenttrader_v15_ab.ps1 points DEFAULT_CONTROL_ROOT/CodexControlRoot at a new path.

Why this exists (found 2026-07-17): a bridge-driven run_v15_ab.py spawns nested child
`codex exec` processes from within the bridge's own sandboxed (workspace-write) codex
process. Reproduced directly: a plain, unsandboxed `codex exec` against a freshly relocated
CODEX_HOME works fine; the identical call made as a NESTED child of a sandboxed parent fails
every time with "os error 10013" (socket access forbidden) trying to reach
wss://api.openai.com/v1/responses -- regardless of --windows-sandbox elevated/unelevated.
The original ~/.codex/ab-v15-v4/arm-a|arm-b paths worked fine as nested children (the v15.4
pilots, t0055/t0056) because they had already been through a one-time interactive elevated
run at some point; the newly relocated .aioptimizer/codex-controls/... paths never were.

This script makes ONE direct (non-nested, non-sandboxed) `codex exec` call per arm, with
Windows sandbox elevation requested explicitly, so whatever OS-level grant Windows needs to
remember for that CODEX_HOME path gets triggered and cached. You WILL see a Windows
elevation/firewall prompt the first time for each arm -- that is the one part no script can
skip; approve it once per arm. After this script completes, nested/bridge-driven calls
against these same CODEX_HOME paths should stop failing with os error 10013 without any
further prompts, matching how the original pilot paths already behaved.

Usage:
    powershell -ExecutionPolicy Bypass -File bootstrap_codex_home_network.ps1
    powershell -ExecutionPolicy Bypass -File bootstrap_codex_home_network.ps1 -ControlRoot "C:\Code\AIOptimizer\.aioptimizer\codex-controls\ab-v15-v4"

Verification, not just a prompt-clearer: each arm's smoke call must return the literal
word OK in its agent_message -- if it errors instead, the grant did not take and this
script says so rather than reporting a false pass.
#>
param(
    [string]$ControlRoot = "C:\Code\AIOptimizer\.aioptimizer\codex-controls\ab-v15-v4",
    [string]$AIOptimizerHome = "C:\Code\AIOptimizer",
    [string]$Model = "gpt-5.6-sol",
    [string]$Codex = "codex.cmd"
)

$ErrorActionPreference = "Stop"

function Test-CodexHomeNetwork {
    param(
        [Parameter(Mandatory = $true)][string]$Arm,
        [Parameter(Mandatory = $true)][string]$CodexHome
    )

    if (-not (Test-Path -LiteralPath $CodexHome)) {
        Write-Host "[$Arm] SKIP: CODEX_HOME not found: $CodexHome"
        return $false
    }

    Write-Host "[$Arm] Running one direct, elevated codex exec against $CodexHome ..."
    Write-Host "[$Arm] Approve any Windows elevation/firewall prompt that appears -- this is the one-time step."

    $env:CODEX_HOME = $CodexHome
    $env:AIOPTIMIZER_HOME = $AIOptimizerHome
    $prompt = "Reply with exactly the word OK and do nothing else."
    $output = $prompt | & $Codex `
        --config 'windows.sandbox="elevated"' `
        --sandbox workspace-write `
        --cd $AIOptimizerHome `
        --model $Model `
        exec --skip-git-repo-check --json - 2>&1
    $exitCode = $LASTEXITCODE
    Remove-Item Env:\CODEX_HOME -ErrorAction SilentlyContinue
    Remove-Item Env:\AIOPTIMIZER_HOME -ErrorAction SilentlyContinue

    $ok = ($exitCode -eq 0) -and ($output -match '"text":"OK"')
    if ($ok) {
        Write-Host "[$Arm] PASS: network-reachable, exit 0, agent replied OK."
    } else {
        Write-Host "[$Arm] FAIL: exit=$exitCode. Full output below."
        Write-Host ($output | Out-String)
    }
    return $ok
}

$armA = Join-Path $ControlRoot "arm-a"
$armB = Join-Path $ControlRoot "arm-b"

$resultA = Test-CodexHomeNetwork -Arm "arm-a" -CodexHome $armA
$resultB = Test-CodexHomeNetwork -Arm "arm-b" -CodexHome $armB

Write-Host ""
if ($resultA -and $resultB) {
    Write-Host "Both arms verified network-reachable. Nested/bridge-driven runs against these CODEX_HOME paths should now work."
    exit 0
} else {
    Write-Host "At least one arm failed. Do not assume the bridge-driven run will work -- rerun this script or investigate before spending real turns."
    exit 1
}
