# Setup Instructions for Claude Code: Environment and Git

Use these steps to prepare the machine for the crab model comparison project. Do not start building the tool until every check at the end passes.

## 1. Check prerequisites

Run these and report the output:

```
python3 --version      # needs 3.10, 3.11 or 3.12
git --version
nvidia-smi             # optional; shows GPU and CUDA version if present
```

- If Python is missing or outside 3.10 to 3.12, stop and tell the user which version to install.
- If git is missing, install it (Windows: git-scm.com; macOS: `xcode-select --install`; Linux: `sudo apt install git`).
- If `nvidia-smi` fails, CPU-only mode is fine for testing. Training all five models will be slow on CPU; tell the user this.

## 2. Set up git

Only run these if the project folder is not already a git repository (`git status` fails).

```
cd crab-compare
git init
git branch -M main
```

Create a `.gitignore` with this content:

```
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
```

Why these are ignored: the image data is large and may be restricted, and model files and results are large generated outputs. Code and configs are tracked; results are reproduced by running the code.

Set the author identity if it is not set globally (ask the user for their name and email, do not invent them):

```
git config user.name "<user name>"
git config user.email "<user email>"
```

Make the first commit after the requirements and handout files exist:

```
git add .gitignore requirements.txt
git commit -m "Initial project setup: requirements and gitignore"
```

Commit after each completed build step (see the build order in the handout). Use short, descriptive messages.

### 2a. Create a GitHub account (done by the user, not by Claude Code)

Claude Code must not create accounts or enter personal credentials for the user. If the user has no GitHub account:

1. The user goes to https://github.com/signup and creates a free account with their own email.
2. They verify the email address.
3. They enable two-factor authentication (Settings → Password and authentication). GitHub requires this for pushing code in many cases.
4. They tell Claude Code their GitHub username.

Then the user creates an empty repository for the project (optional but recommended, so the work is backed up):

1. On GitHub, click **New repository**.
2. Name it, for example `crab-compare`.
3. Set it to **Private**, since the project may contain unpublished research.
4. Leave "Add a README", ".gitignore" and "license" unchecked, because the local project already has them.
5. Copy the repository URL, for example `https://github.com/<username>/crab-compare.git`.

### 2b. Connect the local repository to GitHub

Claude Code runs this once the user provides the URL:

```
git remote add origin https://github.com/<username>/crab-compare.git
git remote -v
```

Authenticate. Choose one option and let the user do the login step:

- **Recommended: GitHub CLI.** Install `gh` (https://cli.github.com), then the user runs `gh auth login` in their own terminal, choosing HTTPS and logging in through the browser. Claude Code then does not need the password or token.
- **Alternative: personal access token.** The user creates one at GitHub → Settings → Developer settings → Personal access tokens, with the `repo` scope, and enters it when git asks for a password on the first push. Claude Code must not write the token into any file or into the git remote URL.
- **Alternative: SSH key.** The user creates a key with `ssh-keygen -t ed25519`, adds the public key under GitHub → Settings → SSH keys, and uses the `git@github.com:` URL form.

Test the connection with an empty push only after the first commit exists:

```
git push -u origin main
```

If the push is refused, report the exact error message. Do not force-push (`git push --force`) without asking the user.

## 3. Create the virtual environment

From the project root:

```
python3 -m venv .venv
```

Activate it:

- macOS / Linux: `source .venv/bin/activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`

Confirm the environment is active: `which python` (or `where python` on Windows) must point inside `.venv`.

## 4. Install PyTorch first

Install PyTorch separately so the correct build is used. Check the GPU result from step 1.

- **NVIDIA GPU:** get the exact command for the local CUDA version from https://pytorch.org/get-started/locally/ and run it inside the activated environment.
- **CPU only:** `pip install torch torchvision`

Then verify:

```
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is False on a machine with an NVIDIA GPU, stop and report it. Do not continue silently on CPU.

## 5. Install the remaining requirements

```
pip install -r requirements.txt
```

If an install fails, report the exact error and the package name. Do not remove packages from `requirements.txt` to make the install pass without asking the user.

## 6. Verify the environment

Run this check script. It must print `OK` for every model:

```
python -c "
import timm, torch
models = {
    'EfficientNet-B3': 'efficientnet_b3',
    'ResNet-50': 'resnet50',
    'DenseNet-121': 'densenet121',
    'MobileNetV3': 'mobilenetv3_large_100',
    'ViT': 'vit_base_patch16_224',
}
for name, arch in models.items():
    m = timm.create_model(arch, pretrained=False, num_classes=2)
    m.eval()
    with torch.no_grad():
        size = timm.data.resolve_data_config({}, model=m)['input_size'][1]
        out = m(torch.randn(1, 3, size, size))
    assert out.shape == (1, 2)
    print(name, 'OK', tuple(out.shape))
import pytorch_grad_cam, onnx, onnxruntime, sklearn, scipy, statsmodels, albumentations, cv2
print('All packages import OK')
"
```

Note: `pretrained=False` is used here only to avoid downloading weights during the check. Pretrained weights are downloaded on first real training run; tell the user this will need internet access.

Also run:

```
git status
```

It should show a clean tree after the first commit.

## 7. Report back

Tell the user:
- Python version, git version, PyTorch version, and whether CUDA is available
- Whether every model passed the verification script
- The output of `git log --oneline` after the first commit
- Any warnings or packages that failed

Do not begin the comparison tool until the user confirms this report.
