"""Keep a running development service safe during release preparation."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows executable sharing rules')
def test_release_environment_rejects_locked_python_before_cleanup(tmp_path):
    script = tmp_path / 'check.ps1'
    script.write_text(r'''
param([string]$BuildScript, [string]$Venv)
$ErrorActionPreference = 'Stop'
$syntax = [System.Management.Automation.Language.Parser]::ParseFile($BuildScript, [ref]$null, [ref]$null)
foreach ($definition in $syntax.FindAll({param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]}, $false)) {
    if ($definition.Name -in @('Fail', 'Assert-ReleaseEnvironmentAvailable')) {
        Invoke-Expression $definition.Extent.Text
    }
}
$context = @{VenvDirectory=$Venv}
Assert-ReleaseEnvironmentAvailable $context
New-Item -ItemType Directory -Force -Path (Join-Path $Venv 'Scripts') | Out-Null
$python = Join-Path $Venv 'Scripts/python.exe'
$sentinel = Join-Path $Venv 'keep.txt'
[IO.File]::WriteAllText($python, 'fixture')
[IO.File]::WriteAllText($sentinel, 'keep')
$lock = [IO.File]::Open($python, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
$rejected = $false
try {
    try { Assert-ReleaseEnvironmentAvailable $context }
    catch {
        if ($_.Exception.Message -notmatch 'choose an unused -ReleaseVenv') { throw }
        $rejected = $true
    }
} finally { $lock.Dispose() }
if (-not $rejected) { throw 'Locked interpreter was accepted' }
if ([IO.File]::ReadAllText($sentinel) -cne 'keep') { throw 'Environment was modified' }
Assert-ReleaseEnvironmentAvailable $context
Write-Output 'PASS release environment guard'
''', encoding='utf-8')
    build_script = Path(__file__).resolve().parents[1] / 'packaging/build_release_windows.ps1'
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-File',
                             str(script), str(build_script), str(tmp_path / 'release env')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS release environment guard' in result.stdout


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows release staging')
def test_native_symbols_are_an_explicit_packaging_option(tmp_path):
    script = tmp_path / 'symbols.ps1'
    script.write_text(r'''
param([string]$BuildScript, [string]$Root)
$ErrorActionPreference = 'Stop'
$syntax = [System.Management.Automation.Language.Parser]::ParseFile($BuildScript, [ref]$null, [ref]$null)
foreach ($definition in $syntax.FindAll({param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]}, $false)) {
    if ($definition.Name -eq 'Stage-WindowsRuntime') { Invoke-Expression $definition.Extent.Text }
}
function Reset-Directory($Path) {
    if (Test-Path $Path) { Remove-Item $Path -Recurse -Force }
    New-Item $Path -ItemType Directory | Out-Null
}
function Fail($Message) { throw $Message }
function Find-CudaRuntimeDirectory($Path) { return $Path }
function Copy-VisualCppRuntimeDlls($Path) {}
function Copy-RuntimeDlls($Source, $Destination, $Patterns = @('*.dll')) {
    foreach ($pattern in $Patterns) {
        Get-ChildItem $Source -Filter $pattern | Copy-Item -Destination $Destination
    }
}
$native = Join-Path $Root 'native'
New-Item $native -ItemType Directory | Out-Null
foreach ($name in @('mfq-runtime.pdb', 'mfq-diagnostics.pdb', 'cudart64_1.dll', 'cublas64_1.dll', 'cublasLt64_1.dll')) {
    Set-Content (Join-Path $native $name) 'fixture'
}
$context = @{NativeOutput=@{DirectoryName=$native}; WindowsRuntimeDirectory=(Join-Path $Root 'stage'); CudaRoot=$native}
$IncludeNativeSymbols = $false
Stage-WindowsRuntime $context
if (Get-ChildItem $context.WindowsRuntimeDirectory -Filter '*.pdb') { throw 'symbols included by default' }
$IncludeNativeSymbols = $true
Stage-WindowsRuntime $context
if (@(Get-ChildItem $context.WindowsRuntimeDirectory -Filter '*.pdb').Count -ne 2) { throw 'symbols missing' }
Write-Output 'PASS optional native symbols'
''', encoding='utf-8')
    build_script = Path(__file__).resolve().parents[1] / 'packaging/build_release_windows.ps1'
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-File',
                             str(script), str(build_script), str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS optional native symbols' in result.stdout
