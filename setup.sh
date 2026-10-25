#!/usr/bin/env bash
# Automated setup for the crab model comparison project.
# Safe to re-run: each step checks whether it is already done.
#
# Usage (from the project folder):
#   bash setup.sh                 # CPU PyTorch build
#   TORCH_INDEX=cu121 bash setup.sh   # NVIDIA GPU, CUDA 12.1 build
#
# Does NOT: log in to GitHub, set git name/email, or push to any remote.

set -euo pipefail

PROJECT_NAME="crab-compare"
PY_MIN_MINOR=10
PY_MAX_MINOR=12
TORCH_INDEX="${TORCH_INDEX:-cpu}"   # cpu, cu121, cu124, ...

say()  { printf '\n==> %s\n' "$*"; }
ok()   { printf '    OK: %s\n' "$*"; }
warn() { printf '    WARN: %s\n' "$*"; }
die()  { printf '    FAIL: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. prerequisites
say "1. Checking prerequisites"

PY=""
for cand in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    minor=$("$cand" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)
    major=$("$cand" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)
    if [ "$major" = "3" ] && [ "$minor" -ge "$PY_MIN_MINOR" ] && [ "$minor" -le "$PY_MAX_MINOR" ]; then
      PY="$cand"; break
    fi
  fi
done
[ -n "$PY" ] || die "Need Python 3.10-3.12. Install it, then re-run."
ok "Python: $($PY --version) ($PY)"

command -v git >/dev/null 2>&1 || die "git not found. Install git, then re-run."
ok "git: $(git --version)"

if command -v nvidia-smi >/dev/null 2>&1; then
  ok "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  warn "No NVIDIA GPU detected. CPU training will be slow."
fi

# ---------------------------------------------------------------- 2. git repository
say "2. Git repository"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  # Refuse to run inside a parent repo, so files are never staged into it.
  # --show-prefix is empty only at the top level; avoids comparing path spellings.
  if [ -n "$(git rev-parse --show-prefix)" ]; then
    die "This folder is inside another git repository ($(git rev-parse --show-toplevel)). Run setup.sh from that repository's root, or move the project out of it."
  fi
  ok "Already a git repository"
else
  git init -q
  git symbolic-ref HEAD refs/heads/main
  ok "Initialised git repository (branch: main)"
fi

if [ ! -f .gitignore ]; then
  cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
data/
results/
checkpoints/
*.pth
*.onnx
*.ckpt
.DS_Store
.ipynb_checkpoints/
EOF
  ok "Created .gitignore"
else
  ok ".gitignore already exists"
fi

GIT_NAME=$(git config user.name || true)
GIT_EMAIL=$(git config user.email || true)
if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ]; then
  warn "Git name/email not set. Run these yourself with your own details:"
  warn '  git config user.name "Your Name"'
  warn '  git config user.email "you@example.com"'
else
  ok "Git identity: $GIT_NAME <$GIT_EMAIL>"
fi

# ---------------------------------------------------------------- 3. virtual environment
say "3. Virtual environment"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  ok "Created .venv"
else
  ok ".venv already exists"
fi

# shellcheck disable=SC1091
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate
else
  die "Could not find the venv activation script"
fi
python -m pip install -q --upgrade pip
ok "Active Python: $(python --version) at $(which python)"

# ---------------------------------------------------------------- 4. PyTorch
say "4. PyTorch ($TORCH_INDEX build)"

install_torch() {
  python -m pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
  ok "Installed torch and torchvision ($TORCH_INDEX)"
}

TORCH_VER=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || true)
TORCH_CUDA=$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)
if [ -z "$TORCH_VER" ]; then
  install_torch
elif [ "$TORCH_INDEX" != "cpu" ] && [ -z "$TORCH_CUDA" ]; then
  warn "torch $TORCH_VER is a CPU-only build but TORCH_INDEX=$TORCH_INDEX was requested. Reinstalling."
  python -m pip uninstall -y -q torch torchvision
  install_torch
else
  ok "torch already installed: $TORCH_VER"
fi

CUDA_AVAIL=$(python -c 'import torch; print(torch.cuda.is_available())')
if [ "$TORCH_INDEX" != "cpu" ] && [ "$CUDA_AVAIL" != "True" ]; then
  die "TORCH_INDEX=$TORCH_INDEX was requested but CUDA is not available. Check the driver or choose the right CUDA version."
fi
ok "CUDA available: $CUDA_AVAIL"

# ---------------------------------------------------------------- 5. project requirements
say "5. Installing requirements.txt"

[ -f requirements.txt ] || die "requirements.txt not found in $(pwd)"
python -m pip install -q -r requirements.txt
ok "Requirements installed"

# ---------------------------------------------------------------- 6. verification
say "6. Verification"

cat > verify_env.py <<'EOF'
import sys
import timm
import torch

models = {
    "EfficientNet-B3": "efficientnet_b3",
    "ResNet-50": "resnet50",
    "DenseNet-121": "densenet121",
    "MobileNetV3": "mobilenetv3_large_100",
    "ViT": "vit_base_patch16_224",
}

failures = []
for name, arch in models.items():
    try:
        m = timm.create_model(arch, pretrained=False, num_classes=2).eval()
        size = timm.data.resolve_data_config({}, model=m)["input_size"][1]
        with torch.no_grad():
            out = m(torch.randn(1, 3, size, size))
        assert tuple(out.shape) == (1, 2), out.shape
        print(f"  OK   {name} ({arch}) output {tuple(out.shape)}")
    except Exception as e:
        failures.append(name)
        print(f"  FAIL {name} ({arch}): {e}")

packages = ["pytorch_grad_cam", "onnx", "onnxruntime", "sklearn", "scipy",
            "statsmodels", "albumentations", "cv2", "pandas", "matplotlib"]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  OK   import {pkg}")
    except Exception as e:
        failures.append(pkg)
        print(f"  FAIL import {pkg}: {e}")

print(f"\nPython {sys.version.split()[0]}, torch {torch.__version__}, "
      f"CUDA {torch.cuda.is_available()}")
sys.exit(1 if failures else 0)
EOF

if python verify_env.py; then
  ok "All models and packages verified"
else
  die "Verification failed. See the FAIL lines above and report them."
fi

# ---------------------------------------------------------------- 7. first commit
say "7. First commit"

mkdir -p data results
COMMIT_FILES=".gitignore .gitattributes requirements.txt setup.sh verify_env.py CLAUDE.md
  handoff.md CLAUDE_CODE_HANDOUT.md CLAUDE_CODE_SETUP.md"
for f in $COMMIT_FILES; do
  if [ -f "$f" ]; then
    git add -- "$f"
  else
    warn "Not staged, file missing: $f"
  fi
done
if git diff --cached --quiet; then
  ok "Nothing new to commit"
elif [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ]; then
  warn "Skipping commit: set git name and email first, then run: git commit -m 'Initial project setup'"
else
  git commit -q -m "Initial project setup: environment, requirements and verification"
  ok "Committed. git log --oneline -1:"
  git log --oneline -1
fi

# ---------------------------------------------------------------- 8. remaining manual steps
say "Done. Remaining steps for you (not automated):"
cat <<EOF
  1. Create the GitHub account and a PRIVATE repository named $PROJECT_NAME.
  2. Log in on this machine: run 'gh auth login' (recommended), or use a token/SSH key.
  3. Connect and push:
       git remote add origin https://github.com/<your-username>/$PROJECT_NAME.git
       git push -u origin main
  4. Tell Claude Code the results of this run before building the comparison tool.
EOF
