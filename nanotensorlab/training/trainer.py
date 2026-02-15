import os
import torch
from torch.utils.data import DataLoader


class Trainer:
    """
    Generic Trainer class.

    The Trainer is fully agnostic to:
    - the type of problem (regression, sampling, diffusion, etc.)
    - the structure of the batch

    The only contract is:
        model.loss(batch) -> torch.Tensor
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config,
        device="cpu",
        output_dir="outputs",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["lr"],
        )

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
        )

    # --------------------------------------------------
    # Core steps
    # --------------------------------------------------

    def _move_batch_to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return tuple(b.to(self.device) for b in batch)
        return batch.to(self.device)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()
            loss = self.model.loss(batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)
            loss = self.model.loss(batch)
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    def fit(self):
        best_val_loss = float("inf")

        for epoch in range(self.config["training"]["epochs"]):

            train_loss = self.train_one_epoch()
            val_loss = self.evaluate()

            print(
                f"[Epoch {epoch+1}/{self.config['training']['epochs']}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
                )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best.pt")

        # Save last checkpoint
        self.save_checkpoint("last.pt")

    # --------------------------------------------------
    # Checkpointing
    # --------------------------------------------------

    def save_checkpoint(self, filename):
        path = os.path.join(self.output_dir, filename)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
