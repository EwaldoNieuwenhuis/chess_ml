<#
.SYNOPSIS
    Copies all Markdown (*.md) files in the repo (respecting .gitignore) to the clipboard.
.EXAMPLE
    .\copy_md_to_clipboard.ps1
#>

$ErrorActionPreference = "Stop"

# Navigate to repo root if inside a git repo
try {
    $repoRoot = (git rev-parse --show-toplevel).Trim()
    Set-Location $repoRoot
} catch {
    # Fallback to current directory
}

Write-Host "📋 Gathering Markdown files (excluding .gitignore)..." -ForegroundColor Cyan

# Gather markdown files respecting .gitignore
$mdFiles = git ls-files --cached --others --exclude-standard "*.md"

if (-not $mdFiles) {
    Write-Warning "No markdown (*.md) files found."
    exit 0
}

$sb = [System.Text.StringBuilder]::new()
$count = 0

foreach ($file in $mdFiles) {
    if (Test-Path $file -PathType Leaf) {
        $count++
        [void]$sb.AppendLine("================================================================================")
        [void]$sb.AppendLine("FILE: $file")
        [void]$sb.AppendLine("================================================================================")
        $content = Get-Content -Path $file -Raw -Encoding UTF8
        [void]$sb.AppendLine($content)
        [void]$sb.AppendLine("`n")
    }
}

$sb.ToString() | Set-Clipboard

Write-Host "✅ Successfully copied $count Markdown file(s) to clipboard!" -ForegroundColor Green
$mdFiles | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
