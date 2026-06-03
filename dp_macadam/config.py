from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    seeds: list[int] = field(default_factory=lambda: [42])
    sigmas: list[float] = field(default_factory=lambda: [0.5])
    batch_size: int = 256
    num_epochs: int = 5
    eta: float = 0.001
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    results_dir: str = "results"
    results_prefix: str = "mnist_results"
    data_root: str = "data"
    num_workers: int = 2

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    h1: float = 1e-9
    beta3: float = 0.999
    gamma: float = 1e-8

