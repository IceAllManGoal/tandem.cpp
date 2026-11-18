#!/usr/bin/env pwsh
#Requires -Version 5.1
<#
.SYNOPSIS
    Build tandem.cpp on Windows x64: MSVC + NVIDIA CUDA, llama-server.exe.

.DESCRIPTION
    Windows counterpart of build.sh, with the same contract:
      1. clone llama.cpp into -Dir if it is not there yet, and check out the commit in BASE_COMMIT
      2. apply tandem.patch (skipped when it is already applied)
      3. configure with CMake (-DGGML_CUDA=ON) and build the llama-server target

    No models are downloaded and no tests or benchmarks are run.
    Needs an NVIDIA driver/CUDA Toolkit with nvcc, and Visual Studio with the C++ toolset.

.PARAMETER Dir
    llama.cpp checkout, created next to this script if missing (default: "tandem").

.PARAMETER CudaArch
    CMAKE_CUDA_ARCHITECTURES, e.g. 89 (RTX 40xx), 120 (Blackwell), or "89;120".
    Default: read from nvidia-smi when it is available, otherwise llama.cpp's own default list,
    which covers more GPUs but takes longer to compile.

.PARAMETER Generator
    Auto (default), Ninja or VS. Auto uses Ninja when ninja.exe and cl.exe are already on PATH
    (an MSVC "Developer PowerShell"), and the Visual Studio generator otherwise.

.PARAMETER CmakeArgs
    Extra arguments for the configure step, e.g.
        .\build.ps1 -CmakeArgs -DGGML_CUDA_FORCE_MMQ=ON
    PowerShell splits an array into separate arguments, so pass them as an array for several:
        .\build.ps1 -CmakeArgs "-DGGML_CUDA_FORCE_MMQ=ON","-DGGML_CUDA_FA_ALL_QUANTS=ON"

.PARAMETER PatchCheckOnly
    Only check that tandem.patch applies to BASE_COMMIT. Nothing is configured or built.

.PARAMETER ConfigureOnly
    Apply the patch and run the CMake configure step, then stop before building.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\build.ps1

.EXAMPLE
    .\build.ps1 -Dir C:\src\llama.cpp -CudaArch 89

.EXAMPLE
    .\build.ps1 -PatchCheckOnly
#>
[CmdletBinding()]
param(
    [string]   $Dir = "tandem",
    [string]   $CudaArch = "",
    [ValidateSet("Auto", "Ninja", "VS")]
    [string]   $Generator = "Auto",
    [string[]] $CmakeArgs = @(),
    [switch]   $PatchCheckOnly,
    [switch]   $ConfigureOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Fail([string] $Message) {
    Write-Host "build.ps1: $Message" -ForegroundColor Red
    exit 1
}

function Step([string] $Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Have([string] $Name) {
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

# "Visual Studio 17 2022" and so on, from the installed toolset rather than a hard-coded name
function Get-VsGenerator {
    $pf86 = ${env:ProgramFiles(x86)}
    if (-not $pf86) {
        Fail "ProgramFiles(x86) is not set: this does not look like a 64-bit Windows"
    }
    $vswhere = Join-Path $pf86 "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Fail "vswhere.exe not found: install Visual Studio 2022 with the 'Desktop development with C++' workload"
    }
    $version = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion
    if (-not $version) {
        Fail "no Visual Studio installation with the C++ toolset found (vswhere reported nothing)"
    }
    $major = [int]($version.Split('.')[0])
    switch ($major) {
        17 { return "Visual Studio 17 2022" }
        16 { return "Visual Studio 16 2019" }
        default {
            Fail "Visual Studio $version is not supported here: build.ps1 knows the 2019 and 2022 generators"
        }
    }
}

$Here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not [System.IO.Path]::IsPathRooted($Dir)) {
    $Dir = Join-Path (Get-Location).Path $Dir
}
$Dir = [System.IO.Path]::GetFullPath($Dir)

$BaseFile = Join-Path $Here "BASE_COMMIT"
$Patch    = Join-Path $Here "tandem.patch"
if (-not (Test-Path $BaseFile)) { Fail "BASE_COMMIT not found next to build.ps1" }
if (-not (Test-Path $Patch))    { Fail "tandem.patch not found next to build.ps1" }
$Base = (Get-Content -Raw $BaseFile).Trim()
if (-not $Base) { Fail "BASE_COMMIT is empty" }

# ------------------------------------------------------------------------------------------------
# toolchain
# ------------------------------------------------------------------------------------------------
Step "checking the toolchain"
if (-not (Have git))   { Fail "git not found on PATH (install Git for Windows)" }
if (-not (Have cmake)) { Fail "cmake not found on PATH (install CMake, or start the 'x64 Native Tools Command Prompt for VS' / a Developer PowerShell that has it)" }

$nvcc = $null
if (Have nvcc) {
    $nvcc = (Get-Command nvcc).Source
} elseif ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
    $nvcc = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
}
if (-not $nvcc) {
    Fail "nvcc not found: install the CUDA Toolkit (12.x/13.x) and set CUDA_PATH, or start a shell with CUDA on PATH"
}
Write-Host "    nvcc:  $nvcc"
Write-Host "    cmake: $((& cmake --version | Select-Object -First 1))"

$gen = $Generator
if ($gen -eq "Auto") {
    # Ninja only makes sense with an MSVC environment already loaded (cl.exe on PATH)
    if ((Have ninja) -and (Have cl)) { $gen = "Ninja" } else { $gen = "VS" }
}
if ($gen -eq "Ninja") {
    $genArgs  = @("-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release")
    $genLabel = "Ninja"
} else {
    $vsGen    = Get-VsGenerator
    $genArgs  = @("-G", $vsGen, "-A", "x64")
    $genLabel = $vsGen
}
Write-Host "    generator: $genLabel"

# ------------------------------------------------------------------------------------------------
# llama.cpp at the pinned commit, with the patch applied
# ------------------------------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $Dir ".git"))) {
    Step "cloning llama.cpp into $Dir"
    & git clone https://github.com/ggml-org/llama.cpp $Dir
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed" }
}

Push-Location $Dir
try {
    Step "checking out $Base"
    & git checkout -q $Base
    if ($LASTEXITCODE -ne 0) { Fail "git checkout $Base failed (is the checkout complete?)" }

    # same three states as build.sh: applies, already applied, does not apply
    & git apply --check $Patch 2>$null
    if ($LASTEXITCODE -eq 0) {
        Step "applying tandem.patch"
        & git apply $Patch
        if ($LASTEXITCODE -ne 0) { Fail "git apply failed" }
    } else {
        & git apply --check --reverse $Patch 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    tandem.patch is already applied"
        } else {
            Fail "tandem.patch does not apply to $(& git rev-parse --short HEAD) (local changes?)"
        }
    }
} finally {
    Pop-Location
}

if ($PatchCheckOnly) {
    Step "tandem.patch applies to $Base"
    exit 0
}

# ------------------------------------------------------------------------------------------------
# configure and build
# ------------------------------------------------------------------------------------------------
$arch = $CudaArch
if (-not $arch -and (Have nvidia-smi)) {
    $cc = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1
    if ($cc -and ($cc.Trim() -match '^(\d+)\.(\d+)$')) {
        $arch = "$($Matches[1])$($Matches[2])"      # "8.9" -> "89", "12.0" -> "120"
    }
}
$archArgs = @()
if ($arch) {
    $archArgs = @("-DCMAKE_CUDA_ARCHITECTURES=$arch")
    Write-Host "    CMAKE_CUDA_ARCHITECTURES: $arch"
} else {
    Write-Host "    CMAKE_CUDA_ARCHITECTURES: llama.cpp default (pass -CudaArch to narrow the build)"
}

$buildDir      = Join-Path $Dir "build"
$cmakeConfigure = @("-S", $Dir, "-B", $buildDir) + $genArgs + @("-DGGML_CUDA=ON") + $archArgs + $CmakeArgs

Step "configuring"
Write-Host ("    cmake " + ($cmakeConfigure -join " "))
& cmake @cmakeConfigure
if ($LASTEXITCODE -ne 0) { Fail "cmake configure failed" }

if ($ConfigureOnly) {
    Step "configured: $buildDir"
    exit 0
}

Step "building llama-server"
& cmake --build $buildDir --config Release --target llama-server --parallel
if ($LASTEXITCODE -ne 0) { Fail "build failed" }

# single-config generators (Ninja) drop the executable in bin\, multi-config ones (VS) in bin\Release\
$cands = @(
    (Join-Path $buildDir "bin\Release\llama-server.exe"),
    (Join-Path $buildDir "bin\llama-server.exe"),
    (Join-Path $buildDir "Release\bin\llama-server.exe")
)
$exe = $null
foreach ($cand in $cands) {
    if (Test-Path $cand) { $exe = $cand; break }
}
if (-not $exe) {
    Fail ("the build finished but llama-server.exe was not found; looked in: " + ($cands -join ", "))
}

Step "built: $exe"
Write-Host ""
Write-Host "Next steps (build.ps1 never downloads models):"
Write-Host "  python .\tandem-run.py --model qwen3.8 --dry-run"
Write-Host "  python .\tandem-run.py --model qwen3.8 --server-bin `"$exe`""
