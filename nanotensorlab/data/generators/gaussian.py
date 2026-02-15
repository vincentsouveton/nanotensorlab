import torch


class GaussianGenerator:
    """
    Isotropic Gaussian generator in arbitrary dimension.

    x ~ N(mean, std^2 I_d)
    """

    def __init__(
        self,
        dim: int = 2,
        mean: float = 0.0,
        std: float = 1.0,
        device: str = "cpu"
    ):
        self.dim = dim
        self.mean = mean
        self.std = std
        self.device = device

    def sample(self, n: int) -> torch.Tensor:
        """
        Generate n samples.

        Return:
            Tensor shape (n, dim)
        """
        x = torch.randn(n, self.dim, device=self.device)
        x = self.mean + self.std * x
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log densité of isotropic Gaussian.
        """
        var = self.std ** 2
        log_norm = -0.5 * self.dim * torch.log(
            torch.tensor(2 * torch.pi * var, device=x.device)
        )
        quad = -0.5 * torch.sum((x - self.mean) ** 2, dim=1) / var
        return log_norm + quad
