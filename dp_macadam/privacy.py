from opacus.accountants import create_accountant


def estimate_epsilon(noise_multiplier: float, sample_rate: float, total_steps: int, delta: float):
    accountant = create_accountant("prv")
    accountant.history = [(noise_multiplier, sample_rate, total_steps)]
    return accountant.get_epsilon(delta=delta)

