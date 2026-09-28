#!/usr/bin/env bash
# One-time environment setup on a RunPod GPU pod (everything lives on /workspace,
# the persistent volume, so it survives pod restarts).
#
#   bash scripts/setup_runpod.sh
#
# Installs Miniforge (conda) into /workspace/miniforge3, creates the "egypocket"
# env (Python 3.11, torch 2.14 + CUDA 12.6), installs the project and registers
# the Jupyter kernel "Python (egypocket)".
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-/workspace/miniforge3}"
ENV_NAME="${ENV_NAME:-egypocket}"

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo ">>> installing Miniforge into $CONDA_DIR"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
  rm /tmp/miniforge.sh
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">>> creating conda env $ENV_NAME"
  conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.11
fi
conda activate "$ENV_NAME"

echo ">>> installing torch 2.14.0 (CUDA 12.6 build: runs on every driver >= 525)"
pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
echo ">>> installing EgyPocket-TTS and its dependencies"
pip install --no-cache-dir -r "$REPO_DIR/requirements.txt"
pip install --no-cache-dir -e "$REPO_DIR" --no-deps

python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python (egypocket)"

python - <<'EOF'
import torch, pocket_tts, transformers, tokenizers, tensorboard
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
print("pocket_tts OK | transformers", transformers.__version__, "| tokenizers", tokenizers.__version__)
EOF

cat <<EOF

Done. In a new terminal:
    source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME
In JupyterLab: pick the kernel "Python (egypocket)" for the notebooks in $REPO_DIR/notebooks.
EOF
