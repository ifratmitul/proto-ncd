#env-setup.sh
#!/bin/bash
set -e

echo "=========================================="
echo "Setting up CB-NCD Environment"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment from YAML
echo ""
echo "Creating conda environment from env.yml..."
conda env create -f env.yml

echo ""
echo "Creating necessary directories..."
mkdir -p logs/wandb checkpoints data

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="

# Verify installation
echo ""
echo "Verifying installation..."
conda run -n CB-NCD python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
conda run -n CB-NCD python -c "import torchvision; print(f'✓ torchvision: {torchvision.__version__}')"
conda run -n CB-NCD python -c "import pytorch_lightning; print(f'✓ PyTorch Lightning: {pytorch_lightning.__version__}')"
conda run -n CB-NCD python -c "import wandb; print(f'✓ Wandb: {wandb.__version__}')"
conda run -n CB-NCD python -c "import sklearn; print(f'✓ scikit-learn: {sklearn.__version__}')"

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
conda run -n CB-NCD python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}'); print(f'✓ CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Activate environment:"
echo "   conda activate CB-NCD"
echo ""
echo ""
echo "3. (Optional) Setup Wandb:"
echo "   wandb login"
echo "   # Or use --offline flag to disable logging"
echo ""
echo ""
echo "Directories created:"
echo "  - logs/wandb/    (Wandb logs)"
echo "  - checkpoints/   (Model checkpoints)"
echo "  - data/          (Datasets)"
echo ""
echo "=========================================="