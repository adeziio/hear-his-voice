#!/bin/bash

set -e

echo "=========================================="
echo "       Hear His Voice Linux Installer"
echo "=========================================="

echo
echo "Checking Python..."

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python 3 not found."
    exit 1
fi

python3 --version

echo
echo "Upgrading pip..."

python3 -m pip install --upgrade pip

echo
echo "Installing PyTorch..."

python3 -m pip install --upgrade \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo
echo "Installing Hear His Voice dependencies..."

python3 -m pip install --upgrade -r requirements.txt

echo
echo "Checking FFmpeg..."

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: FFmpeg not found."
    echo "Please install FFmpeg before continuing."
    exit 1
else
    echo "FFmpeg detected."
fi

echo
echo "Checking Ollama..."

if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed."
fi

echo
echo "Stopping any existing Ollama process..."

pkill -f "ollama serve" >/dev/null 2>&1 || true
pkill -f "ollama runner" >/dev/null 2>&1 || true
pkill -f "ollama" >/dev/null 2>&1 || true

sleep 2

echo
echo "Starting Ollama in CPU-only mode..."

OLLAMA_LOG="/tmp/hhv-ollama.log"

env \
    CUDA_VISIBLE_DEVICES="" \
    NVIDIA_VISIBLE_DEVICES="" \
    OLLAMA_VULKAN=0 \
    OLLAMA_NUM_GPU=0 \
    nohup ollama serve > "$OLLAMA_LOG" 2>&1 &

echo
echo "Waiting for Ollama..."

OLLAMA_READY=false

for i in {1..30}; do

    if curl -s --max-time 2 \
        http://localhost:11434/api/tags \
        >/dev/null 2>&1; then

        OLLAMA_READY=true
        break

    fi

    sleep 1

done

if [ "$OLLAMA_READY" != true ]; then

    echo
    echo "ERROR: Ollama failed to start."

    echo
    echo "Ollama log:"
    echo "------------------------------------------"

    cat "$OLLAMA_LOG" 2>/dev/null || true

    echo "------------------------------------------"

    exit 1

fi

echo "Ollama is ready."

echo
echo "Checking Qwen model..."

if ! ollama list | grep -q "qwen3:8b"; then

    echo "qwen3:8b not found."
    echo "Downloading qwen3:8b..."

    ollama pull qwen3:8b

else

    echo "qwen3:8b already installed."

fi

echo
echo "Verifying PyTorch and CUDA..."

python3 -c "
import torch

print('PyTorch:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA Device:', torch.cuda.get_device_name(0))
else:
    print('Device: CPU')
"

echo
echo "Checking GPU memory..."

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

echo
echo "=========================================="
echo "       Installation Complete"
echo "=========================================="

echo
echo "Run Hear His Voice with:"
echo
echo "bash ./run_linux.sh"
echo