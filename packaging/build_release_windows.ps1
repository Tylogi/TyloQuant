# Build the Windows x64 MFQ Studio installer, single-file app, and CUDA sidecar.
#
# The release Python environment is isolated from development environments. It
# contains only the base dependencies needed by `mfq serve`; training,
# Calibration, quantization, MiniCPM-o, and PyTorch are not packaged.
# The script imports the Visual Studio environment and adds all required tools to
# its own PATH, so it can be run from ordinary PowerShell.

#requires -Version 7
[CmdletBinding()]
param(
    [string] $PythonVersion = $env:MFQ_RELEASE_PYTHON,
    [string] $ReleaseVenv = $env:MFQ_RELEASE_VENV,
    [string] $OutputDirectory = $env:MFQ_RELEASE_OUTPUT_DIR,
    [string] $CudaArchitectures = $env:MFQ_RELEASE_CUDA_ARCHITECTURES,
    [int] $Jobs = 0,
    [switch] $PortableOnly,
    [switch] $IncludeNativeSymbols,
    [switch] $Clean
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
        $installDirectory = & $vswhere -all -products * -property installationPath |
            Where-Object {
                Test-Path -LiteralPath (Join-Path $_ "Common7\Tools\VsDevCmd.bat") `
                    -PathType Leaf
            } |
            Select-Object -First 1
    }
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

    # CMake may enable OpenMP for mfq-runtime when the VS workload provides it.
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
        [Parameter(Mandatory)][string] $Output,
        [string] $ScriptDirectory = $PSScriptRoot
    )

    $projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $scriptDirectory ".."))
    $resourceDirectory = Join-Path $scriptDirectory "Resources"
    $sidecarDirectory = Join-Path $scriptDirectory "sidecars"
    $targetTriple = "x86_64-pc-windows-msvc"
    $cliName = "mfq-cli"

    return @{
        ScriptDirectory = $scriptDirectory
        PathChecker = Join-Path $scriptDirectory "check_release_paths.py"
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
        NativeArtifact = Join-Path $sidecarDirectory "mfq-runtime-$targetTriple.exe"
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
    # Tauri downloads its pinned NSIS toolset when no local installation exists.
    Initialize-ReleasePathMapping $Context
}

function Get-ReleasePathMappings {
    param([Parameter(Mandatory)][hashtable] $Context)

    $mappings = @(
        [pscustomobject]@{ Source = $Context.ProjectDirectory; Destination = "/mfq-src" }
        [pscustomobject]@{ Source = $env:USERPROFILE; Destination = "/mfq-build/home" }
    )
    if (-not [string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
        $mappings += [pscustomobject]@{
            Source = Resolve-BuildPath $env:CARGO_TARGET_DIR $Context.StudioDirectory
            Destination = "/mfq-build/target"
        }
    }
    if (-not ('MfqRelease.BuildPaths' -as [type])) {
        Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
using System.Text;
namespace MfqRelease {
    public static class BuildPaths {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        public static extern uint GetShortPathName(string path, StringBuilder output, uint size);
    }
}
'@
    }
    foreach ($mapping in $mappings) {
        $mapping
        # Native dependency generators can use Windows 8.3 path aliases.
        $shortPath = [System.Text.StringBuilder]::new(32768)
        $length = [MfqRelease.BuildPaths]::GetShortPathName($mapping.Source, $shortPath, $shortPath.Capacity)
        if ($length -gt 0 -and $length -lt $shortPath.Capacity -and $shortPath.ToString() -ne $mapping.Source) {
            [pscustomobject]@{ Source = $shortPath.ToString(); Destination = $mapping.Destination }
        }
    }
}

function Get-ReleaseNativeFlags {
    param([Parameter(Mandatory)][hashtable] $Context)

    "/experimental:deterministic"
    foreach ($mapping in @(Get-ReleasePathMappings $Context)) {
        foreach ($source in @($mapping.Source, $mapping.Source.Replace('\', '/')) | Select-Object -Unique) {
            '"/pathmap:' + $source + '=' + $mapping.Destination + '"'
        }
    }
}

function Initialize-ReleasePathMapping {
    param([Parameter(Mandatory)][hashtable] $Context)

    # Cargo's encoded form keeps a path containing spaces in a single argument.
    $separator = [char]31
    $rustArguments = @()
    if ($env:CARGO_ENCODED_RUSTFLAGS) {
        $rustArguments = @($env:CARGO_ENCODED_RUSTFLAGS.Split($separator))
    } elseif ($env:RUSTFLAGS) {
        $rustArguments = @($env:RUSTFLAGS -split '\s+' | Where-Object { $_ })
    }
    foreach ($mapping in @(Get-ReleasePathMappings $Context)) {
        foreach ($source in @($mapping.Source, $mapping.Source.Replace('\', '/')) | Select-Object -Unique) {
            $rustArguments += "--remap-path-prefix=$source=$($mapping.Destination)"
        }
    }
    $rustArguments += @("-C", "link-arg=/PDBALTPATH:mfq-studio.pdb")
    $env:CARGO_ENCODED_RUSTFLAGS = $rustArguments -join $separator
    # Native dependencies built by Rust also expand __FILE__.
    $nativeFlags = @(Get-ReleaseNativeFlags $Context) -join ' '
    $env:CFLAGS = "$env:CFLAGS $nativeFlags".Trim()
    $env:CXXFLAGS = "$env:CXXFLAGS $nativeFlags".Trim()
    $env:CC_SHELL_ESCAPED_FLAGS = "1"
}

function Assert-ReleasePaths {
    param(
        [Parameter(Mandatory)][hashtable] $Context,
        [Parameter(Mandatory)][string[]] $Paths,
        [switch] $PythonArchive
    )

    $arguments = @($Context.PathChecker) + $Paths
    foreach ($mapping in @(Get-ReleasePathMappings $Context)) {
        $arguments += @("--private-prefix", $mapping.Source)
    }
    if ($PythonArchive) { $arguments += @("--python-cli", $Context.CliArtifact) }
    Invoke-Checked $Context.Python $arguments
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
    $temporaryDirectory = Join-Path $Context.ScriptDirectory "build\tmp"
    New-Item -ItemType Directory -Force -Path $temporaryDirectory | Out-Null
    $env:TEMP = $temporaryDirectory
    $env:TMP = $temporaryDirectory
    $env:PYINSTALLER_CONFIG_DIR = Join-Path $Context.ScriptDirectory "build\pyinstaller-cache"
}

function Stage-LicenseNotices {
    param([Parameter(Mandatory)][hashtable] $Context)

    $licenseDirectory = Join-Path $Context.ResourceDirectory "licenses"
    Reset-Directory $licenseDirectory
    Copy-Item -LiteralPath (Join-Path $Context.ProjectDirectory "LICENSE") `
        -Destination (Join-Path $licenseDirectory "MFQ-Apache-2.0.txt") -Force
    Copy-Item -LiteralPath (Join-Path $Context.ProjectDirectory "NOTICE") `
        -Destination (Join-Path $licenseDirectory "NOTICE.txt") -Force
    Get-ChildItem -LiteralPath (Join-Path $Context.ProjectDirectory "LICENSES") `
        -File -Filter "*.txt" |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $licenseDirectory -Force
        }
}

function Assert-ReleaseEnvironmentAvailable {
    param([Parameter(Mandatory)][hashtable] $Context)

    $python = Join-Path $Context.VenvDirectory "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        return
    }
    $handle = $null
    try {
        $handle = [System.IO.File]::Open(
            $python, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    }
    catch [System.IO.IOException], [System.UnauthorizedAccessException] {
        Fail "release Python environment is in use or not writable; choose an unused -ReleaseVenv: $($Context.VenvDirectory)"
    }
    finally {
        if ($null -ne $handle) { $handle.Dispose() }
    }
}

function Initialize-PythonEnvironment {
    param(
        [Parameter(Mandatory)][hashtable] $Context,
        [Parameter(Mandatory)][string] $Version
    )

    Assert-ReleaseEnvironmentAvailable $Context
    Invoke-Checked $Context.Tools.Uv @("venv", "--clear", "--python", $Version, $Context.VenvDirectory)
    $env:VIRTUAL_ENV = $Context.VenvDirectory
    Add-ToPath (Join-Path $Context.VenvDirectory "Scripts")

    $syncArguments = @(
        "sync", "--locked", "--active", "--no-default-groups", "--no-editable",
        "--group", "release", "--extra", "benchmark"
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

    # mfq-runtime uses the native tensor backend and must remain independent of
    # Python and LibTorch. Explicitly disable the optional A/B reference target
    # so a reused CMake cache cannot pull those dependencies back into a release.
    $nativeFlags = @(Get-ReleaseNativeFlags $Context)
    $cudaFlags = ($nativeFlags | ForEach-Object { "--compiler-options=$_" }) -join ' '
    Invoke-Checked $Context.Tools.CMake @(
        "-S", (Join-Path $Context.ProjectDirectory "cpp_runtime"),
        "-B", $Context.NativeBuildDirectory,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_FLAGS=$env:CFLAGS",
        "-DCMAKE_CXX_FLAGS=$env:CXXFLAGS",
        "-DCMAKE_CUDA_FLAGS=$env:CUDAFLAGS $cudaFlags",
        "-DCMAKE_CUDA_COMPILER=$([System.IO.Path]::Combine($Context.CudaRoot, 'bin', 'nvcc.exe'))",
        "-DCMAKE_CUDA_ARCHITECTURES=$Architectures",
        "-DMFQ_CUDA_ARCHITECTURES=$Architectures",
        "-DMFQ_BUILD_RUNTIME_COMMUNICATION=ON",
        "-DMFQ_BUILD_METAL_RUNTIME=OFF",
        "-DMFQ_BUILD_TORCH_REFERENCE_RUNTIME=OFF"
    )
    Invoke-Checked $Context.Tools.CMake @(
        "--build", $Context.NativeBuildDirectory,
        "--target", "mfq-runtime", "mfq-diagnostics",
        "--config", "Release",
        "--parallel", $ParallelJobs
    )

    $Context.NativeOutput = Get-ChildItem -LiteralPath $Context.NativeBuildDirectory `
        -Recurse -File -Filter "mfq-runtime.exe" |
        Sort-Object -Property LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $Context.NativeOutput) {
        Fail "native build did not create mfq-runtime.exe"
    }
    Copy-Item -LiteralPath $Context.NativeOutput.FullName -Destination $Context.NativeArtifact -Force
    $diagnostics = Get-ChildItem -LiteralPath $Context.NativeBuildDirectory `
        -Recurse -File -Filter "mfq-diagnostics.exe" |
        Sort-Object -Property LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $diagnostics) {
        Fail "native build did not create mfq-diagnostics.exe"
    }
    Copy-Item -LiteralPath $diagnostics.FullName -Destination `
        (Join-Path $Context.SidecarDirectory "mfq-diagnostics-$($Context.TargetTriple).exe") -Force
}

function Stage-WindowsRuntime {
    param([Parameter(Mandatory)][hashtable] $Context)

    Reset-Directory $Context.WindowsRuntimeDirectory
    if ($IncludeNativeSymbols) {
        foreach ($name in @("mfq-runtime.pdb", "mfq-diagnostics.pdb")) {
            Copy-Item -LiteralPath (Join-Path $Context.NativeOutput.DirectoryName $name) `
                -Destination $Context.WindowsRuntimeDirectory -Force
        }
    }
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
            "huggingface_hub", "modelscope_hub", "uvicorn", "websockets", "pyarrow",
            "pandas", "sympy", "regex", "loguru", "antlr4", "yaml", "tokenizers")) {
        $arguments += @("--collect-all", $package)
    }
    foreach ($module in @(
            "torch", "transformers", "safetensors",
            "scipy", "tiktoken",
            "mfq.calibration", "mfq.quantize", "mfq.runtime", "mfq.tools",
            "mfq.runtime.minicpmo45", "mfq.runtime.minicpmo45_realtime",
            "minicpmo_utils", "stepaudio2_minicpmo", "torchaudio", "onnxruntime",
            "s3tokenizer", "hyperpyyaml", "librosa")) {
        $arguments += @("--exclude-module", $module)
    }
    $arguments += @(
        "--copy-metadata", "huggingface-hub",
        "--copy-metadata", "modelscope-hub"
    )
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

    $cargoTargetDirectory = if ([string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
        Join-Path $Context.StudioDirectory "src-tauri\target"
    } else {
        Resolve-BuildPath $env:CARGO_TARGET_DIR $Context.StudioDirectory
    }
    $releaseDirectory = Join-Path $cargoTargetDirectory "$($Context.TargetTriple)\release"
    Push-Location $Context.StudioDirectory
    try {
        Invoke-Checked $Context.Tools.Npm @("ci")
        Invoke-Checked $Context.Tools.Npm @("run", "check")
        $env:CI = "true"
        Invoke-Checked $Context.Tools.Npm @(
            "run", "tauri", "--", "build",
            "--config", "src-tauri/tauri.release-windows.conf.json",
            "--target", $Context.TargetTriple,
            "--no-bundle"
        )
        Assert-ReleasePaths $Context @(
            (Join-Path $releaseDirectory "mfq-studio.exe"),
            $Context.SidecarDirectory, $Context.ResourceDirectory,
            $Context.WindowsRuntimeDirectory, (Join-Path $Context.StudioDirectory "dist")
        ) -PythonArchive
        Invoke-Checked $Context.Tools.Npm @(
            "run", "tauri", "--", "bundle",
            "--config", "src-tauri/tauri.release-windows.conf.json",
            "--target", $Context.TargetTriple,
            "--bundles", "nsis"
        )
    }
    finally {
        Pop-Location
    }

    $nsisDirectory = Join-Path $cargoTargetDirectory `
        "$($Context.TargetTriple)\release\bundle\nsis"
    $installer = Get-ChildItem -LiteralPath $nsisDirectory -File -Filter "*.exe" `
        -ErrorAction SilentlyContinue |
        Sort-Object -Property LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $installer) {
        Fail "Tauri did not create an NSIS installer"
    }

    $releaseInstaller = Join-Path $Context.OutputDirectory $installer.Name
    Assert-ReleasePaths $Context @($installer.FullName)
    Copy-Item -LiteralPath $installer.FullName -Destination $releaseInstaller -Force
    Write-Host "copied $releaseInstaller"
}

function Build-PortableArchive {
    param([Parameter(Mandatory)][hashtable] $Context)

    $cargoTargetDirectory = if ([string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
        Join-Path $Context.StudioDirectory "src-tauri\target"
    } else {
        Resolve-BuildPath $env:CARGO_TARGET_DIR $Context.StudioDirectory
    }
    $executable = Join-Path $cargoTargetDirectory "$($Context.TargetTriple)\release\mfq-studio.exe"
    $makensis = Join-Path $cargoTargetDirectory ".tauri\NSIS\makensis.exe"
    if (-not (Test-Path -LiteralPath $makensis -PathType Leaf)) {
        $makensis = Require-Command "makensis"
    }
    Assert-ReleasePaths $Context @(
        $executable, $Context.SidecarDirectory, $Context.ResourceDirectory,
        $Context.WindowsRuntimeDirectory, (Join-Path $Context.StudioDirectory "dist")
    ) -PythonArchive
    Invoke-Checked $Context.Python @(
        (Join-Path $Context.ScriptDirectory "build_portable_windows.py"),
        "--studio-executable", $executable,
        "--output-directory", $Context.OutputDirectory,
        "--makensis", $makensis
    )
}

if (-not [System.OperatingSystem]::IsWindows() -or -not [Environment]::Is64BitOperatingSystem) {
    Fail "this release target must be built on 64-bit Windows"
}

if ($Clean -and $PortableOnly) {
    Fail "-PortableOnly reuses the existing build and cannot be combined with -Clean"
}

if ($Clean) {
    foreach ($name in @("build", "sidecars", "Frameworks", "Resources", "windows-runtime", "dist")) {
        $path = Join-Path $PSScriptRoot $name
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
}

$PythonVersion = if ([string]::IsNullOrWhiteSpace($PythonVersion)) { "3.12" } else { $PythonVersion }
$CudaArchitectures = if ([string]::IsNullOrWhiteSpace($CudaArchitectures)) { "80-real;86-real;89" } else { $CudaArchitectures }
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
if ($PortableOnly) {
    $context.Python = Join-Path $context.VenvDirectory "Scripts\python.exe"
    Build-PortableArchive $context
    exit 0
}
Initialize-BuildEnvironment $context
Initialize-ReleaseDirectories $context
Stage-LicenseNotices $context

Push-Location $context.ProjectDirectory
try {
    Initialize-PythonEnvironment $context $PythonVersion
    Build-NativeSidecar $context $CudaArchitectures $Jobs
    Stage-WindowsRuntime $context
    Build-PythonCli $context
    Remove-PyInstallerCache $context
    Build-NsisInstaller $context
    Build-PortableArchive $context
}
finally {
    Pop-Location
}
