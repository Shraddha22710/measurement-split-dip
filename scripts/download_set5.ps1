$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path data\set5 | Out-Null
curl.exe -L https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz -o data\set5\Set5_HR.tar.gz
tar -xzf data\set5\Set5_HR.tar.gz -C data\set5

Write-Host "Downloaded Set5 HR images to data\set5\Set5_HR"
