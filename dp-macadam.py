#!/usr/bin/env python3
"""
DP-MACADAM MNIST experiments. 
Requires GPU for per-sample gradient computations.
"""

# ── Library Check / Install ───────────────────────────────────────────────────
import importlib
import subprocess
import sys

def check_or_install(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name}")
    except ImportError:
        print(f"✗ {package_name} not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        print(f"✓ {package_name} installed")

required = [
    ("torch",        "torch"),
    ("torchvision",  "torchvision"),
    ("opacus",       "opacus"),
    ("numpy",        "numpy"),
    ("tqdm",         "tqdm"),
]

print("Checking required libraries...\n")
for pip_name, import_name in required:
    check_or_install(pip_name, import_name)
print("\nAll libraries ready.\n")

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import gc
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from opacus import GradSampleModule
from opacus.validators import ModuleValidator
from opacus.accountants import create_accountant
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
# seeds         = [42, 123, 456, 789, 999]
# sigmas        = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
seeds         = [42]
sigmas        = [0.5]

BATCH_SIZE    = 256
NUM_EPOCHS    = 5
ETA           = 0.001

NOISE_MULT    = 0.5
MAX_GRAD_NORM = 1.0
DELTA         = 1e-5
ROOT          = "mnist_results"

# Adam hyperparameters
ADAM_BETA1    = 0.9
ADAM_BETA2    = 0.999

# DP-MACADAM hyperparameters
H1_           = 1e-9
BETA3         = 0.999
GAMMA         = 1e-8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Seed ──────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Dataset ───────────────────────────────────────────────────────────────────
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]))
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

print(f"\nTrain size:    {len(train_dataset):,}")
print(f"Test size:     {len(test_dataset):,}")
print(f"Batch size:    {BATCH_SIZE}")
print(f"Batches/epoch: {len(train_loader)}")
print(f"Seeds:         {seeds}")
print(f"Epochs:        {NUM_EPOCHS}")

# ── Privacy Accounting ────────────────────────────────────────────────────────
SAMPLE_RATE = BATCH_SIZE / len(train_dataset)
T_total     = NUM_EPOCHS * len(train_loader)
accountant  = create_accountant("prv")
accountant.history = [(NOISE_MULT, SAMPLE_RATE, T_total)]
eps = accountant.get_epsilon(delta=DELTA)
print(f"\nPrivacy: ε ≈ {eps:.2f}, δ = {DELTA} (Connect-the-Dots / PRV)\n")

# ── Model ─────────────────────────────────────────────────────────────────────
class AdaClipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

def make_fresh_model():
    m = AdaClipNet().to(device)
    m.train()
    return m

print(f"Model parameters: {sum(p.numel() for p in AdaClipNet().parameters()):,}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def free_memory():
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

def save_results(results_to_add, filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            results = json.load(f)
    else:
        results = {}
    results.update(results_to_add)
    with open(filepath, "w") as f:
        json.dump(results, f)
    print(f"Saved. Keys in file: {list(results.keys())}")

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss   = loss_fn(logits, y_batch)
            total_loss    += loss.item() * len(x_batch)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += len(x_batch)
    model.train()
    return total_loss / total_samples, total_correct / total_samples

def make_grad_sample_model(model):
    model.train()
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"Fixing model compatibility: {errors}")
        model = ModuleValidator.fix(model)
    return GradSampleModule(model)

def get_per_sample_grads(gs_model, x_batch, y_batch, loss_fn):
    gs_model.zero_grad()
    out  = gs_model(x_batch)
    loss = loss_fn(out, y_batch)
    loss.backward()
    G = torch.cat([
        p.grad_sample.flatten(start_dim=1)
        for p in gs_model.parameters()
        if p.grad_sample is not None
    ], dim=1)
    for p in gs_model.parameters():
        p.grad_sample = None
    return G

def apply_flat_update(model, update, eta):
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p -= eta * update[idx:idx + numel].reshape(p.shape)
            idx += numel

loss_fn = nn.CrossEntropyLoss()

# ── DP-MACADAM ────────────────────────────────────────────────────────────────
def run_dpmacadam(seed):
    set_seed(seed)
    model    = make_fresh_model()
    gs_model = make_grad_sample_model(model)
    d        = sum(p.numel() for p in model.parameters())
    m  = torch.zeros(d, device=device)
    u  = torch.zeros(d, device=device)
    s2 = torch.zeros(d, device=device)
    b  = torch.full((d,), MAX_GRAD_NORM / d, device=device)
    m_hat_prev = torch.zeros(d, device=device)
    eval_accs = []
    t = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        for x_batch, y_batch in tqdm(train_loader, desc=f"  DP-MACAdam-v3 epoch {epoch}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            B = len(x_batch)
            t += 1
            noise_scale = NOISE_MULT * MAX_GRAD_NORM / B
            G = get_per_sample_grads(gs_model, x_batch, y_batch, loss_fn)
            W = (G - m_hat_prev) / b
            norms = torch.norm(W, dim=1, keepdim=True)
            W_bar = W / torch.clamp(norms, min=1.0)
            g_tilde = b * (W_bar.mean(dim=0) +
                      torch.randn(d, device=device) * noise_scale) \
                      + m_hat_prev
            m = ADAM_BETA1 * m + (1 - ADAM_BETA1) * g_tilde
            u = ADAM_BETA2 * u + (1 - ADAM_BETA2) * g_tilde ** 2
            m_hat  = m / (1 - ADAM_BETA1 ** t)
            u_hat  = u / (1 - ADAM_BETA2 ** t)
            update = m_hat / (torch.sqrt(u_hat) + GAMMA)
            apply_flat_update(model, update, ETA)
            v      = (g_tilde - m_hat_prev) ** 2 - b ** 2 * noise_scale ** 2
            s2     = BETA3 * s2 + (1 - BETA3) * v
            s2_hat = torch.clamp(s2 / (1 - BETA3 ** t), min=H1_)
            s      = torch.sqrt(s2_hat)
            b      = torch.sqrt(s) * torch.sqrt(s.sum())
            m_hat_prev = m_hat.detach().clone()

        _, acc = evaluate(model, test_loader, loss_fn, device)
        eval_accs.append(acc)
        print(f"  Epoch {epoch} | Acc: {acc:.4f}")

    free_memory()
    return eval_accs

# ── Run All ───────────────────────────────────────────────────────────────────
algorithms = {
    "dp-macadam":    run_dpmacadam,
}

for sigma in sigmas:
    print(f"\nNoise scale: {sigma}")
    NOISE_MULT = sigma
    RESULTS_PATH = f"{ROOT}_sigma{str(sigma).replace('.', '_')}.json"

    for algo_name, algo_fn in algorithms.items():
        print(f"\n{'='*60}")
        print(f"Algorithm: {algo_name}")
        print(f"{'='*60}")
        all_accs = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            accs = algo_fn(seed)
            all_accs.append(accs)
        save_results({algo_name: all_accs}, RESULTS_PATH)
        free_memory()

    print("\n" + "="*60)
    print("All experiments complete.")
    print(f"Results saved to: {RESULTS_PATH}")
    print("="*60)

for results_file in [f"{ROOT}_sigma{str(s).replace('.', '_')}.json" for s in sigmas]:
    if not os.path.exists(results_file):
        continue
    with open(results_file, "r") as f:
        results = json.load(f)
    print(f"\n{results_file}")
    print(f"{'Algorithm':<25} {'Final Acc (mean ± std)':>25}")
    print("-" * 52)
    for algo, all_seeds in results.items():
        final_accs = [seed_accs[-1] for seed_accs in all_seeds]
        mean = np.mean(final_accs)
        std  = np.std(final_accs)
        print(f"{algo:<25} {mean:.4f} ± {std:.4f}")