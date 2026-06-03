import torch
from opacus import GradSampleModule
from opacus.validators import ModuleValidator


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)

            total_loss += loss.item() * len(x_batch)
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
    out = gs_model(x_batch)
    loss = loss_fn(out, y_batch)
    loss.backward()

    grads = torch.cat([
        p.grad_sample.flatten(start_dim=1)
        for p in gs_model.parameters()
        if p.grad_sample is not None
    ], dim=1)

    for p in gs_model.parameters():
        p.grad_sample = None

    return grads


def apply_flat_update(model, update, eta):
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p -= eta * update[idx:idx + numel].reshape(p.shape)
            idx += numel

