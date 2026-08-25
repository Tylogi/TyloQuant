# Build the Windows x64 MFQ Studio serve-only NSIS installer and CUDA sidecar.
#
# The release Python environment is isolated from development environments. It
# contains only the daemon dependencies needed by `mfq serve`; training,
# calibration, quantization, TPQ, MiniCPM-o, and PyTorch are not packaged.
# The script imports the Visual Studio environment and adds all required tools to
# its own PATH, so it can be run from ordinary PowerShell.

#requires -Version 7
[CmdletBinding()]
param(
    [string] $PythonVersion = $env:MFQ_RELEASE_PYTHON,
    [string] $ReleaseVenv = $env:MFQ_RELEASE_VENV,
    [string] $OutputDirectory = $env:MFQ_RELEASE_OUTPUT_DIR,
    [string] $CudaArchitectures = $env:MFQ_RELEASE_CUDA_ARCHITECTURES,
    [int] $Jobs = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Fail {
    param([Parameter(Mandatory)][string] $Message)

    throw "error: $Message"
}

function Resolve-BuildPath {
    param(
        [Parameter(Mandatory)][string] $Path,
        [Parameter(Mandatory)][string] $BaseDirectory
    )

    if ([System.IO.Path]::IsPathFullyQualified($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDirectory $Path))
}

function Add-ToPath {
    param([Parameter(Mandatory)][string] $Directory)

    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        return
    }
    if (($env:PATH -split [System.IO.Path]::PathSeparator) -notcontains $Directory) {
        $env:PATH = "$Directory$([System.IO.Path]::PathSeparator)$env:PATH"
    }
}

function Require-Command {
    param([Parameter(Mandatory)][string] $Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $command) {
        Fail "$Name is required; install it and make it available to this release script"
    }
    return $command.Source
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory)][string] $FilePath,
        [string[]] $Arguments = @()
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

function Import-VisualStudioEnvironment {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        Fail "Visual Studio Build Tools with the C++ workload are required"
    }

    $installDirectory = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($installDirectory)) {
        Fail "Visual Studio C++ x64 tools are required"
    }

    $vsDevCmd = Join-Path $installDirectory.Trim() "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $vsDevCmd -PathType Leaf)) {
        Fail "VsDevCmd.bat is missing from $installDirectory"
    }

    $environmentLines = & cmd.exe /d /s /c "call `"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
    if ($LASTEXITCODE -ne 0) {
        Fail "failed to initialize the Visual Studio x64 environment"
    }
    foreach ($line in $environmentLines) {
        $separator = $line.IndexOf('=')
        if ($separator -le 0) {
            continue
        }
        Set-Item -Path "Env:$($line.Substring(0, $separator))" -Value $line.Substring($separator + 1)
    }
}

function Find-CudaRoot {
    if (-not [string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
        if (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH "bin\nvcc.exe") -PathType Leaf) {
            return $env:CUDA_PATH
        }
    }

    $cudaParent = Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $cudaParent -PathType Container) {
        $cudaDirectory = Get-ChildItem -LiteralPath $cudaParent -Directory |
            Sort-Object -Property Name -Descending |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "bin\nvcc.exe") -PathType Leaf } |
            Select-Object -First 1
        if ($null -ne $cudaDirectory) {
            return $cudaDirectory.FullName
        }
    }
    Fail "CUDA Toolkit with nvcc is required"
}

function Find-CudaRuntimeDirectory {
    param([Parameter(Mandatory)][string] $CudaRoot)

    foreach ($candidate in @(
            (Join-Path $CudaRoot "bin\x64"),
            (Join-Path $CudaRoot "bin"))) {
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            $runtime = Get-ChildItem -LiteralPath $candidate -File `
                -Filter "cudart64_*.dll" -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($null -ne $runtime) {
                return $candidate
            }
        }
    }
    Fail "the CUDA Toolkit runtime DLL directory is missing from $CudaRoot"
}

function Copy-RuntimeDlls {
    param(
        [Parameter(Mandatory)][string] $SourceDirectory,
        [Parameter(Mandatory)][string] $DestinationDirectory,
        [string[]] $Patterns = @("*.dll")
    )

    if (-not (Test-Path -LiteralPath $SourceDirectory -PathType Container)) {
        return
    }
    foreach ($pattern in $Patterns) {
        Get-ChildItem -LiteralPath $SourceDirectory -File -Filter $pattern -ErrorAction SilentlyContinue |
            ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination $DestinationDirectory -Force
            }
    }
}

function Copy-VisualCppRuntimeDlls {
    param([Parameter(Mandatory)][string] $DestinationDirectory)

    if ([string]::IsNullOrWhiteSpace($env:VCToolsRedistDir)) {
        Fail "the Visual Studio environment did not provide VCToolsRedistDir"
    }
    $x64Directory = Join-Path $env:VCToolsRedistDir "x64"
    if (-not (Test-Path -LiteralPath $x64Directory -PathType Container)) {
        Fail "the Visual C++ x64 redistributable directory is missing from $env:VCToolsRedistDir"
    }

    $crtDirectory = Get-ChildItem -LiteralPath $x64Directory -Directory |
        Where-Object { $_.Name -match '^Microsoft\.VC\d+\.CRT$' } |
        Select-Object -First 1
    if ($null -eq $crtDirectory) {
        Fail "the Visual C++ x64 CRT redistributable is missing from $x64Directory"
    }
    Copy-RuntimeDlls $crtDirectory.FullName $DestinationDirectory

    # CMake may enable OpenMP for mfq-decode when the VS workload provides it.
    # Staging the runtime when present is harmless if OpenMP was not selected.
    Get-ChildItem -LiteralPath $x64Directory -Directory |
        Where-Object { $_.Name -match '^Microsoft\.VC\d+\.OpenMP$' } |
        ForEach-Object {
            Copy-RuntimeDlls $_.FullName $DestinationDirectory
        }
}

function Reset-Directory {
    param([Parameter(Mandatory)][string] $Path)

    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Get-ReleaseJobs {
    param([int] $RequestedJobs)

    if ($RequestedJobs -gt 0) {
        return $RequestedJobs
    }
    if ($env:MFQ_RELEASE_JOBS) {
        return [int] $env:MFQ_RELEASE_JOBS
    }

    # The largest native CUDA translation units need several GiB each, so bound
    # parallelism by installed memory.
    $memoryJobs = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 5GB)
    return [math]::Max(1, [math]::Min([int] $memoryJobs, [Environment]::ProcessorCount))
}

function New-BuildContext {
    param(
        [Parameter(Mandatory)][string] $Venv,
        [Parameter(Mandatory)][string] $Output
    )

    $scriptDirectory = $PSScriptRoot
    $projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $scriptDirectory ".."))
    $resourceDirectory = Join-Path $scriptDirectory "Resources"
    $sidecarDirectory = Join-Path $scriptDirectory "sidecars"
    $targetTriple = "x86_64-pc-windows-msvc"
    $cliName = "mfq-cli"

    return @{
        ScriptDirectory = $scriptDirectory
        ProjectDirectory = $projectDirectory
        StudioDirectory = Join-Path $projectDirectory "MFQStudio"
        PyInstallerBuildDirectory = Join-Path $scriptDirectory "build\pyinstaller"
        SpecDirectory = Join-Path $scriptDirectory "build\spec"
        NativeBuildDirectory = Join-Path $scriptDirectory "build\runtime"
        SidecarDirectory = $sidecarDirectory
        ResourceDirectory = $resourceDirectory
        WindowsRuntimeDirectory = Join-Path $scriptDirectory "windows-runtime"
        TargetTriple = $targetTriple
        CliName = $cliName
        CliArtifact = Join-Path (Join-Path $resourceDirectory $cliName) "$cliName.exe"
        NativeArtifact = Join-Path $sidecarDirectory "mfq-decode-$targetTriple.exe"
        VenvDirectory = $Venv
        OutputDirectory = $Output
        CudaRoot = $null
        Python = $null
        PyInstaller = $null
        NativeOutput = $null
        Tools = @{ Uv = $null; CMake = $null; Npm = $null }
    }
}

function Initialize-BuildEnvironment {
    param([Parameter(Mandatory)][hashtable] $Context)

    Add-ToPath (Join-Path ${env:ProgramFiles} "CMake\bin")
    Add-ToPath (Join-Path $env:USERPROFILE ".cargo\bin")
    Add-ToPath ([System.IO.Path]::Combine(
        [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFilesX86), "NSIS"))
    Import-VisualStudioEnvironment

    $Context.CudaRoot = Find-CudaRoot
    $env:CUDA_PATH = $Context.CudaRoot
    Add-ToPath (Join-Path $Context.CudaRoot "bin")
    Add-ToPath (Join-Path $Context.CudaRoot "lib\x64")

    $Context.Tools.Uv = Require-Command "uv"
    $Context.Tools.CMake = Require-Command "cmake"
    $null = Require-Command "ninja"
    $null = Require-Command "cargo"
    $Context.Tools.Npm = Require-Command "npm.cmd"
    $null = Require-Command "cl"
    $null = Require-Command "nvcc"
    $null = Require-Command "makensis"
}

function Initialize-ReleaseDirectories {
    param([Parameter(Mandatory)][hashtable] $Context)

    $directories = @(
        $Context.PyInstallerBuildDirectory,
        $Context.SpecDirectory,
        $Context.NativeBuildDirectory,
        $Context.SidecarDirectory,
        $Context.ResourceDirectory,
        $Context.OutputDirectory
    )
    New-Item -ItemType Directory -Force -Path $directories | Out-Null
}

function Initialize-PythonEnvironment {
    param(
        [Parameter(Mandatory)][hashtable] $Context,
        [Parameter(Mandatory)][string] $Version
    )

    Invoke-Checked $Context.Tools.Uv @("venv", "--clear", "--python", $Version, $Context.VenvDirectory)
    $env:VIRTUAL_ENV = $Context.VenvDirectory
    Add-ToPath (Join-Path $Context.VenvDirectory "Scripts")

    $syncArguments = @(
        "sync", "--locked", "--active", "--no-default-groups", "--no-editable",
        "--group", "release", "--extra", "daemon"
    )
    Invoke-Checked $Context.Tools.Uv $syncArguments

    $Context.Python = Join-Path $Context.VenvDirectory "Scripts\python.exe"
    $Context.PyInstaller = Join-Path $Context.VenvDirectory "Scripts\pyinstaller.exe"
    if (-not (Test-Path -LiteralPath $Context.Python -PathType Leaf)) {
        Fail "uv did not create the release Python interpreter"
    }
    if (-not (Test-Path -LiteralPath $Context.PyInstaller -PathType Leaf)) {
        Fail "uv did not install PyInstaller into the release environment"
    }
}

function Build-NativeSidecar {
    param(
        [Parameter(Mandatory)][hashtable] $Context,
        [Parameter(Mandatory)][string] $Architectures,
        [Parameter(Mandatory)][int] $ParallelJobs
    )

    # mfq-decode uses the native tensor backend and must remain independent of
    # Python and LibTorch. Explicitly disable the optional A/B reference target
    # so a reused CMake cache cannot pull those dependencies back into a release.
    Invoke-Checked $Context.Tools.CMake @(
        "-S", (Join-Path $Context.ProjectDirectory "cpp_runtime"),
        "-B", $Context.NativeBuildDirectory,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_COMPILER=$([System.IO.Path]::Combine($Context.CudaRoot, 'bin', 'nvcc.exe'))",
        "-DCMAKE_CUDA_ARCHITECTURES=$Architectures",
        "-DMFQ_CUDA_ARCHITECTURES=$Architectures",
        "-DGGML_NATIVE=OFF",
        "-DMFQ_BUILD_CPP_SERVER=ON",
        "-DMFQ_BUILD_METAL_RUNTIME=OFF",
        "-DMFQ_BUILD_TORCH_REFERENCE_RUNTIME=OFF"
    )
    Invoke-Checked $Context.Tools.CMake @(
        "--build", $Context.NativeBuildDirectory,
        "--target", "mfq-decode",
        "--config", "Release",
        "--parallel", $ParallelJobs
    )

    $Context.NativeOutput = Get-ChildItem -LiteralPath $Context.NativeBuildDirectory `
        -Recurse -File -Filter "mfq-decode.exe" |
        Sort-Object -Property LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $Context.NativeOutput) {
        Fail "native build did not create mfq-decode.exe"
    }
    Copy-Item -LiteralPath $Context.NativeOutput.FullName -Destination $Context.NativeArtifact -Force
}

function Stage-WindowsRuntime {
    param([Parameter(Mandatory)][hashtable] $Context)

    Reset-Directory $Context.WindowsRuntimeDirectory
    Copy-RuntimeDlls $Context.NativeOutput.DirectoryName $Context.WindowsRuntimeDirectory
    $cudaRuntimeDirectory = Find-CudaRuntimeDirectory $Context.CudaRoot
    foreach ($pattern in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")) {
        Copy-RuntimeDlls $cudaRuntimeDirectory $Context.WindowsRuntimeDirectory @($pattern)
        if (-not (Get-ChildItem -LiteralPath $Context.WindowsRuntimeDirectory `
                -File -Filter $pattern -ErrorAction SilentlyContinue)) {
            Fail "the CUDA Toolkit did not provide the native runtime DLL matching $pattern"
        }
    }
    Copy-VisualCppRuntimeDlls $Context.WindowsRuntimeDirectory
    if (-not (Get-ChildItem -LiteralPath $Context.WindowsRuntimeDirectory `
            -File -Filter "*.dll" -ErrorAction SilentlyContinue)) {
        Fail "no native runtime DLLs were staged"
    }
}

function Build-PythonCli {
    param([Parameter(Mandatory)][hashtable] $Context)

    $arguments = @(
        "--noconfirm", "--clean", "--onedir",
        "--name", $Context.CliName,
        "--paths", $Context.ProjectDirectory,
        "--collect-data", "mfq"
    )
    foreach ($package in @(
            "av", "fastapi", "httpx", "PIL", "pydantic", "pypdf",
            "uvicorn", "websockets")) {
        $arguments += @("--collect-all", $package)
    }
    foreach ($module in @(
            "torch", "transformers", "tokenizers", "safetensors",
            "scipy", "pyarrow", "huggingface_hub", "tiktoken",
            "mfq.calibration", "mfq.quantize", "mfq.runtime", "mfq.tools",
            "mfq._vendor.tpq",
            "mfq.runtime.minicpmo45", "mfq.runtime.minicpmo45_realtime",
            "minicpmo_utils", "stepaudio2_minicpmo", "torchaudio", "onnxruntime",
            "s3tokenizer", "hyperpyyaml", "librosa")) {
        $arguments += @("--exclude-module", $module)
    }
    $arguments += @(
        "--distpath", $Context.ResourceDirectory,
        "--workpath", $Context.PyInstallerBuildDirectory,
        "--specpath", $Context.SpecDirectory,
        (Join-Path $Context.ProjectDirectory "mfq\cli.py")
    )
    Invoke-Checked $Context.PyInstaller $arguments

    if (-not (Test-Path -LiteralPath $Context.CliArtifact -PathType Leaf)) {
        Fail "PyInstaller did not create $($Context.CliArtifact)"
    }
    Invoke-Checked $Context.CliArtifact @("--version")
    Invoke-Checked $Context.CliArtifact @("serve", "--help")
    Write-Host "built $($Context.CliArtifact)"
    Write-Host "built $($Context.NativeArtifact)"
}

function Remove-PyInstallerCache {
    param([Parameter(Mandatory)][hashtable] $Context)

    # The workpath is rebuilt by --clean. Reclaim it before Tauri copies the
    # multi-GiB sidecars and runtime resources into its NSIS staging directory.
    if (Test-Path -LiteralPath $Context.PyInstallerBuildDirectory) {
        Remove-Item -LiteralPath $Context.PyInstallerBuildDirectory -Recurse -Force
    }
}

function Build-NsisInstaller {
    param([Parameter(Mandatory)][hashtable] $Context)

    Push-Location $Context.StudioDirectory
    try {
        Invoke-Checked $Context.Tools.Npm @("ci")
        Invoke-Checked $Context.Tools.Npm @("run", "check")
        $env:CI = "true"
        Invoke-Checked $Context.Tools.Npm @(
            "run", "tauri", "--", "build",
            "--config", "src-tauri/tauri.release-windows.conf.json",
            "--target", $Context.TargetTriple,
            "--bundles", "nsis"
        )
    }
    finally {
        Pop-Location
    }

    $nsisDirectory = Join-Path $Context.StudioDirectory `
        "src-tauri\target\$($Context.TargetTriple)\release\bundle\nsis"
    $installer = Get-ChildItem -LiteralPath $nsisDirectory -File -Filter "*.exe" `
        -ErrorAction SilentlyContinue |
        Sort-Object -Property LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $installer) {
        Fail "Tauri did not create an NSIS installer"
    }

    $releaseInstaller = Join-Path $Context.OutputDirectory $installer.Name
    Copy-Item -LiteralPath $installer.FullName -Destination $releaseInstaller -Force
    Write-Host "copied $releaseInstaller"
}

if (-not [System.OperatingSystem]::IsWindows() -or -not [Environment]::Is64BitOperatingSystem) {
    Fail "this release target must be built on 64-bit Windows"
}

$PythonVersion = if ([string]::IsNullOrWhiteSpace($PythonVersion)) { "3.12" } else { $PythonVersion }
$CudaArchitectures = if ([string]::IsNullOrWhiteSpace($CudaArchitectures)) { "86" } else { $CudaArchitectures }
$Jobs = Get-ReleaseJobs $Jobs

$venvDirectory = if ([string]::IsNullOrWhiteSpace($ReleaseVenv)) {
    Join-Path $PSScriptRoot ".venv"
} else {
    Resolve-BuildPath $ReleaseVenv $PSScriptRoot
}
$releaseOutputDirectory = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    Join-Path $PSScriptRoot "dist"
} else {
    Resolve-BuildPath $OutputDirectory $PSScriptRoot
}

$context = New-BuildContext $venvDirectory $releaseOutputDirectory
Initialize-BuildEnvironment $context
Initialize-ReleaseDirectories $context

Push-Location $context.ProjectDirectory
try {
    Initialize-PythonEnvironment $context $PythonVersion
    Build-NativeSidecar $context $CudaArchitectures $Jobs
    Stage-WindowsRuntime $context
    Build-PythonCli $context
    Remove-PyInstallerCache $context
    Build-NsisInstaller $context
}
finally {
    Pop-Location
}
