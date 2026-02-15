import torch
import torch.nn as nn


class AffineFlow(nn.Module):
    """
    Simple diagonal affine normalizing flow.

    z ~ N(0, I)
    x = scale * z + shift
    """

    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.shift = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))

    # --------------------------------------------------
    # Forward transform
    # --------------------------------------------------

    def forward(self, z):
        scale = torch.exp(self.log_scale)
        return scale * z + self.shift

    # --------------------------------------------------
    # Inverse transform
    # --------------------------------------------------

    def inverse(self, x):
        scale = torch.exp(self.log_scale)
        return (x - self.shift) / scale

    # --------------------------------------------------
    # Log likelihood
    # --------------------------------------------------

    def log_prob(self, x):

        z = self.inverse(x)

        # log p(z)
        log_base = -0.5 * torch.sum(z ** 2, dim=-1)
        log_base -= 0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi))

        # log |det J|
        log_det = -torch.sum(self.log_scale)

        return log_base + log_det

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    def loss(self, batch):

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        log_prob = self.log_prob(x)
        return -log_prob.mean()

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------

    @torch.no_grad()
    def sample(self, n):

        z = torch.randn(n, self.dim, device=self.shift.device)
        return self.forward(z)
