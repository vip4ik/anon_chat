# Замена console.log -> log в App.js
$file = "src\App.js"
$content = Get-Content $file -Raw
$content = $content -replace 'console\.log\(', 'log('
$content = $content -replace 'console\.warn\(', 'warn('
$content = $content -replace 'console\.info\(', 'info('
Set-Content $file -Value $content -NoNewline
Write-Host "✅ Заменено!"