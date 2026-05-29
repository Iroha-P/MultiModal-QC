param(
    [switch]$NoDownload,
    [switch]$NoLaunch,
    [switch]$CpuOnly,
    [switch]$NoHfMirror
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvPip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
$ReleaseCache = Join-Path $ProjectRoot "release\downloads"
$RepoRelease = "https://github.com/Iroha-P/MultiModal-QC/releases/download/v1.0.0"

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Warn($Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Find-Python {
    $candidates = @(
        @("py", "-3.11"),
        @("py", "-3"),
        @("python")
    )
    foreach ($candidate in $candidates) {
        $cmd = $candidate[0]
        $args = @()
        if ($candidate.Length -gt 1) {
            $args = $candidate[1..($candidate.Length - 1)]
        }
        try {
            $version = & $cmd @args -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ([version]$version -ge [version]"3.10") {
                return @{ Command = $cmd; Args = $args }
            }
        } catch {
        }
    }
    return $null
}

function Install-Python-IfNeeded {
    $python = Find-Python
    if ($python) {
        return $python
    }

    Write-Warn "Python 3.10+ was not found."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Step "Installing Python 3.11 with winget"
        winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
        $python = Find-Python
        if ($python) {
            return $python
        }
    }

    throw "Please install Python 3.10+ manually, then run oneclick.bat again: https://www.python.org/downloads/"
}

function Invoke-Python($Python, [string[]]$Arguments) {
    & $Python.Command @($Python.Args + $Arguments)
}

function Download-Asset($Name) {
    New-Item -ItemType Directory -Force -Path $ReleaseCache | Out-Null
    $target = Join-Path $ReleaseCache $Name
    if (Test-Path $target) {
        Write-Host "Using cached $Name"
        return $target
    }
    $url = "$RepoRelease/$Name"
    Write-Step "Downloading $Name"
    Invoke-WebRequest -Uri $url -OutFile $target
    return $target
}

function Expand-Zip($ZipPath) {
    Write-Step "Extracting $(Split-Path $ZipPath -Leaf)"
    Expand-Archive -Path $ZipPath -DestinationPath $ProjectRoot -Force
}

Set-Location $ProjectRoot

Write-Step "Checking Python"
$python = Install-Python-IfNeeded

if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating virtual environment"
    Invoke-Python $python @("-m", "venv", ".venv")
}

Write-Step "Upgrading pip"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

$torchIndex = "https://download.pytorch.org/whl/cpu"
if (-not $CpuOnly -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $torchIndex = "https://download.pytorch.org/whl/cu121"
}

Write-Step "Installing PyTorch"
& $VenvPython -m pip install torch torchvision --index-url $torchIndex
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install PyTorch from $torchIndex."
}

Write-Step "Installing project dependencies"
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Default pip install failed. Retrying with Tsinghua mirror."
    & $VenvPython -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install project dependencies."
    }
}

New-Item -ItemType Directory -Force -Path "storage", "outputs\test_detection_data", "models" | Out-Null

if (-not $NoDownload) {
    $assets = @(
        "MultiModal-QC-dataset-metadata.zip",
        "MultiModal-QC-dataset-mvtec-sample.zip",
        "MultiModal-QC-lora-best.zip",
        "MultiModal-QC-baselines.zip"
    )
    foreach ($asset in $assets) {
        try {
            Expand-Zip (Download-Asset $asset)
        } catch {
            Write-Warn "Failed to download/extract ${asset}: $($_.Exception.Message)"
        }
    }

    if (-not (Test-Path "models\Qwen2-VL-2B-Instruct\config.json")) {
        Write-Step "Downloading Qwen2-VL-2B-Instruct"
        if (-not $NoHfMirror -and -not $env:HF_ENDPOINT) {
            $env:HF_ENDPOINT = "https://hf-mirror.com"
            Write-Host "Using HF_ENDPOINT=$env:HF_ENDPOINT"
        }
        try {
            & $VenvPython -m pip install huggingface_hub
            & (Join-Path $ProjectRoot ".venv\Scripts\huggingface-cli.exe") download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen2-VL-2B-Instruct
        } catch {
            Write-Warn "Base model download failed. You can still start the UI, but Qwen inspection will be unavailable until the model is placed in models\Qwen2-VL-2B-Instruct."
        }
    }
}

Write-Step "Setup complete"
Write-Host "Project root: $ProjectRoot"
Write-Host "Gradio UI:   http://127.0.0.1:7860"
Write-Host "API docs:    http://127.0.0.1:8000/docs"

if (-not $NoLaunch) {
    Write-Step "Starting services"
    & (Join-Path $ProjectRoot "start.bat")
}
