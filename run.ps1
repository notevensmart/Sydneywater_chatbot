# Model credentials are read exclusively from the project's .env by the application.
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.env')) { throw 'Missing project .env file.' }
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.runtime'
$env:UV_SYSTEM_CERTS = 'true'
$env:PYTHONUTF8 = '1'
# Preserve certificate verification using the Windows trusted roots for all clients.
if (-not $env:SSL_CERT_FILE -and (Test-Path '.venv\Scripts\python.exe')) {
    New-Item -ItemType Directory -Force '.cache' | Out-Null
    & '.venv\Scripts\python.exe' -c "import ssl, pathlib; roots=[ssl.DER_cert_to_PEM_cert(c) for store in ('ROOT','CA') for c,e,t in ssl.enum_certificates(store) if e=='x509_asn' and (t is True or ssl.Purpose.SERVER_AUTH.oid in t)]; pathlib.Path('.cache/windows-ca.pem').write_text(''.join(roots),encoding='ascii')"
    $env:SSL_CERT_FILE = Join-Path $PSScriptRoot '.cache\windows-ca.pem'
    $env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
}
$taskUv = if (Test-Path '.bootstrap\bin\uv.exe') { '.bootstrap\bin\uv.exe' } else { 'uv' }
if (-not $CommandArgs) { throw 'Pass a command, for example: .\run.ps1 python knowledge.py status' }
& $taskUv run --frozen --extra groq --env-file .env @CommandArgs
exit $LASTEXITCODE
