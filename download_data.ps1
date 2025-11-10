# MovieLens 20M Dataset Download Script (PowerShell)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  MovieLens 20M Dataset Downloader" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Downloading MovieLens 20M dataset..." -ForegroundColor Yellow
Write-Host "Note: Dataset size is ~190MB, download may take a few minutes" -ForegroundColor Yellow
Write-Host ""

# Create data directory
$DataDir = "data"
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir | Out-Null
    Write-Host "Created directory: $DataDir" -ForegroundColor Green
}

Set-Location $DataDir

# Download dataset
$Url = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
$ZipFile = "ml-20m.zip"

Write-Host "Downloading from GroupLens..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $Url -OutFile $ZipFile
    Write-Host "Download complete!" -ForegroundColor Green
} catch {
    Write-Host "Error downloading dataset: $_" -ForegroundColor Red
    exit 1
}

# Extract
Write-Host ""
Write-Host "Extracting dataset..." -ForegroundColor Yellow
Expand-Archive -Path $ZipFile -DestinationPath . -Force

# Cleanup
Write-Host "Cleaning up..." -ForegroundColor Yellow
Remove-Item $ZipFile

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "  Download Complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "Dataset location: $(Join-Path (Get-Location) 'ml-20m')" -ForegroundColor Cyan
Write-Host "Files:" -ForegroundColor Cyan
Write-Host "  - ratings.csv (20M ratings)" -ForegroundColor Cyan
Write-Host "  - movies.csv (27K movies)" -ForegroundColor Cyan
Write-Host ""
