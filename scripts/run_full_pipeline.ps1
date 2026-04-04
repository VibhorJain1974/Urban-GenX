# Urban-GenX | Full Pipeline Runner (PowerShell)
# Run: .\scripts\run_full_pipeline.ps1

Write-Host "=" * 60
Write-Host "  Urban-GenX | Full Pipeline Runner"
Write-Host "=" * 60

# Step 1: Create directories
Write-Host "`n[1/8] Creating directories..."
New-Item -ItemType Directory -Force -Path data\raw\cityscapes, data\raw\urbansound8k, data\raw\metr-la, data\raw\usgs_water, checkpoints | Out-Null

# Step 2: Download datasets (except Cityscapes)
Write-Host "`n[2/8] Downloading datasets..."
python src\utils\download_datasets.py

# Step 3: Download water data
Write-Host "`n[3/8] Downloading USGS water data..."
python src\utils\download_water_data.py

# Step 4: Run tests
Write-Host "`n[4/8] Running unit tests..."
python tests\test_models.py

# Step 5: Train Vision (DP-GAN)
Write-Host "`n[5/8] Training Vision cGAN (DP-SGD)..."
python src\training\train_vision.py

# Step 6: Train Acoustic (VAE)
Write-Host "`n[6/8] Training Acoustic VAE..."
python src\training\train_acoustic.py

# Step 7: Train Traffic + Water (Utility VAE)
Write-Host "`n[7/8] Training Traffic VAE..."
python src\training\train_utility.py

Write-Host "`nTraining Water VAE..."
python scripts\train_water.py

# Step 8: Run Federated Learning
Write-Host "`n[8/8] Running Federated Learning simulation..."
python src\federated\server.py

# Done
Write-Host "`n[DONE] Full pipeline complete!"
Write-Host "Run dashboard: streamlit run dashboard\app.py"
Write-Host "Run MIA audit: python src\utils\privacy_audit.py --model all"
