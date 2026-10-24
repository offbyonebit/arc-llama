$ValidationRoot = $PSScriptRoot
$env:APPDATA = Join-Path $ValidationRoot 'appdata'
$env:LOCALAPPDATA = Join-Path $ValidationRoot 'localappdata'
$env:HF_HOME = Join-Path $ValidationRoot 'hf-cache'
$env:UV_CACHE_DIR = Join-Path $ValidationRoot 'uv-cache'
$env:PIP_CACHE_DIR = Join-Path $ValidationRoot 'pip-cache'
$env:TEMP = Join-Path $ValidationRoot 'temp'
$env:TMP = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
$env:PATH = (Join-Path $ValidationRoot 'venv\Scripts') + ';' + $env:PATH
New-Item -ItemType Directory -Force $env:APPDATA,$env:LOCALAPPDATA,$env:HF_HOME,$env:UV_CACHE_DIR,$env:PIP_CACHE_DIR,$env:TEMP | Out-Null
