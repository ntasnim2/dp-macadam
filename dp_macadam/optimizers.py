import torch
from tqdm import tqdm

from .model import make_fresh_model
from .train import (
    apply_flat_update,
    evaluate,
    get_per_sample_grads,
    make_grad_sample_model,
)
from .utils import free_memory, set_seed


def run_dpmacadam(seed, sigma, config, train_loader, test_loader, loss_fn, device):
    set_seed(seed)

    model = make_fresh_model(device)
    gs_model = make_grad_sample_model(model)
    d = sum(p.numel() for p in model.parameters())

    m = torch.zeros(d, device=device)
    u = torch.zeros(d, device=device)
    s2 = torch.zeros(d, device=device)
    b = torch.full((d,), config.max_grad_norm / d, device=device)
    m_hat_prev = torch.zeros(d, device=device)

    eval_accs = []
    t = 0

    for epoch in range(1, config.num_epochs + 1):
        for x_batch, y_batch in tqdm(train_loader, desc=f"  DP-MACAdam epoch {epoch}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            batch_size = len(x_batch)
            t += 1

            noise_scale = sigma * config.max_grad_norm / batch_size
            gradients = get_per_sample_grads(gs_model, x_batch, y_batch, loss_fn)

            scaled = (gradients - m_hat_prev) / b
            norms = torch.norm(scaled, dim=1, keepdim=True)
            clipped = scaled / torch.clamp(norms, min=1.0)

            noisy_grad = b * (
                clipped.mean(dim=0) + torch.randn(d, device=device) * noise_scale
            ) + m_hat_prev

            m = config.adam_beta1 * m + (1 - config.adam_beta1) * noisy_grad
            u = config.adam_beta2 * u + (1 - config.adam_beta2) * noisy_grad ** 2

            m_hat = m / (1 - config.adam_beta1 ** t)
            u_hat = u / (1 - config.adam_beta2 ** t)
            update = m_hat / (torch.sqrt(u_hat) + config.gamma)
            apply_flat_update(model, update, config.eta)

            variance_estimate = (noisy_grad - m_hat_prev) ** 2
            s2 = config.adam_beta1 * s2 + (1 - config.adam_beta1) * variance_estimate
            kappa = 2 * (config.adam_beta1 - config.adam_beta1 ** t) / (1 + config.adam_beta1)
            s2_hat = torch.clamp(s2 / kappa - b ** 2 * noise_scale ** 2, min=config.h1, max=config.h2)
            s = torch.sqrt(s2_hat)
            b = torch.sqrt(s) * torch.sqrt(s.sum())

            m_hat_prev = m_hat.detach().clone()

        _, acc = evaluate(model, test_loader, loss_fn, device)
        eval_accs.append(acc)
        print(f"  Epoch {epoch} | Acc: {acc:.4f}")

    free_memory(device)
    return eval_accs

