import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.sampler import create_training_sample


class SASRecDataset(Dataset):
    """
    Dataset adapter for training SASRec.
    
    Wraps the user training interaction history dictionary and samples positive/negative 
    items on-the-fly using the project's existing sampler logic.
    """
    def __init__(self, user_train: dict, item_count: int, maxlen: int):
        self.user_train = user_train
        self.user_ids = list(user_train.keys())
        self.item_count = item_count
        self.maxlen = maxlen

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict:
        user_id = self.user_ids[idx]
        user_sequence = self.user_train[user_id]
        
        # Sample training sequence using the project's existing sampler function
        sample = create_training_sample(user_id, user_sequence, self.item_count, self.maxlen)
        
        # Convert all training sample lists to standard PyTorch tensors
        return {
            "user": torch.tensor(sample["user"], dtype=torch.long),
            "sequence": torch.tensor(sample["sequence"], dtype=torch.long),
            "positive": torch.tensor(sample["positive"], dtype=torch.long),
            "negative": torch.tensor(sample["negative"], dtype=torch.long)
        }


class SASRecTrainer:
    """
    Trainer for the SASRec recommender model.

    Maintains the training loops, optimizers, and scoring functions.
    It isolates training objectives and optimizer states from the SASRec model layers.
    """
    def __init__(self, model: nn.Module, lr: float = 0.001, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        
        # Adam optimizer with beta2=0.98 aligned with the reference implementation
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            betas=(0.9, 0.98)
        )

    def calculate_loss(
        self, 
        seq_rep: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the masked Binary Cross Entropy (BCE) loss for sequential recommendation.

        Args:
            seq_rep: Contextual sequence representations, shape (batch_size, maxlen, hidden_units)
            positive: Positive targets (next-item IDs), shape (batch_size, maxlen)
            negative: Negative targets (sampled item IDs), shape (batch_size, maxlen)

        Returns:
            Scalar PyTorch loss tensor
        """
        # Lookup positive and negative item embeddings using the model's learned weights
        # Shape: (batch_size, maxlen, hidden_units)
        pos_emb = self.model.item_emb(positive)
        neg_emb = self.model.item_emb(negative)

        # Compute dot product between representation and positive/negative items
        # Shape: (batch_size, maxlen)
        pos_logits = (seq_rep * pos_emb).sum(dim=-1)
        neg_logits = (seq_rep * neg_emb).sum(dim=-1)

        # BCE loss with logits for positive targets (target label = 1) and negative targets (target label = 0)
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction="none")
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits), reduction="none")

        # Combine losses
        loss = pos_loss + neg_loss

        # Mask padding positions: where positive target == 0 is padding (not a valid interaction target)
        istarget = (positive != 0).float()
        loss = loss * istarget

        # Compute average loss over valid items only (avoid division by zero via a tiny epsilon if necessary)
        num_targets = istarget.sum()
        if num_targets > 0:
            loss = loss.sum() / num_targets
        else:
            loss = loss.sum()

        return loss

    def train_step(self, batch: dict) -> float:
        """
        Performs one gradient descent step over a batch of data.

        Args:
            batch: Dictionary containing sequence, positive, and negative tensors

        Returns:
            Scalar loss float value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Unpack tensors and move them to the training device
        sequence = batch["sequence"].to(self.device)
        positive = batch["positive"].to(self.device)
        negative = batch["negative"].to(self.device)

        # Forward pass through SASRec
        seq_rep = self.model(sequence)

        # Compute masked BCE loss
        loss = self.calculate_loss(seq_rep, positive, negative)

        # Backpropagation and optimizer step
        loss.backward()
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    from src.models.sasrec import SASRec

    print("Running Trainer Sanity Checks...")

    # 1. Imports work
    print("[1/8] Imports work: Passed")

    # 2. Setup dummy model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SASRec(
        item_count=100,
        maxlen=5,
        hidden_units=50,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.5
    )

    trainer = SASRecTrainer(model=model, lr=0.001, device=device)
    print("[2/8] Trainer instantiates: Passed")

    # 3. Create a synthetic batch
    batch = {
        "user": torch.tensor([1, 2], dtype=torch.long),
        "sequence": torch.tensor([
            [0, 0, 10, 20, 30],
            [0, 5, 15, 25, 35]
        ], dtype=torch.long),
        "positive": torch.tensor([
            [0, 0, 20, 30, 40],
            [0, 15, 25, 35, 45]
        ], dtype=torch.long),
        "negative": torch.tensor([
            [0, 0, 99, 98, 97],
            [0, 96, 95, 94, 93]
        ], dtype=torch.long),
    }

    # Check if a single training step runs
    loss_1 = trainer.train_step(batch)
    print(f"[3/8] One training step completes: Passed (Loss: {loss_1:.4f})")

    # 4. Check if loss is finite and not NaN
    assert not math.isnan(loss_1), "Loss is NaN!"
    assert math.isfinite(loss_1), "Loss is not finite!"
    print("[4/8] Loss is finite and not NaN: Passed")

    # 5. Check if gradients are produced
    grad = model.item_emb.weight.grad
    assert grad is not None, "No gradients produced for item embedding weights!"
    assert torch.any(grad != 0), "Gradients are all zeros!"
    print("[5/8] Gradients are produced and non-zero: Passed")

    # Save a copy of parameter to check for updates
    weight_before = model.item_emb.weight.clone()

    # 6. Run a second step
    loss_2 = trainer.train_step(batch)
    print(f"[6/8] A second step runs without errors: Passed (Loss: {loss_2:.4f})")

    # 7. Check if loss does not become NaN
    assert not math.isnan(loss_2), "Loss became NaN in second step!"
    print("[7/8] Loss does not become NaN in subsequent step: Passed")

    # 8. Optimizer updates model parameters
    weight_after = model.item_emb.weight.clone()
    assert not torch.allclose(weight_before, weight_after), "Model parameters were not updated by optimizer!"
    print("[8/8] Optimizer updates model parameters: Passed")

    print("\nALL TRAINER SANITY TESTS PASSED SUCCESSFULLY!")
