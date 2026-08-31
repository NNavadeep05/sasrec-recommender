import os
import sys
import argparse
import random
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to sys.path to guarantee import safety from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import load_dataset
from src.models.sasrec import SASRec
from src.training.trainer import SASRecDataset, SASRecTrainer


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility where possible.
    
    Note: SASRec negative sampling is CPU-based random and may differ run-to-run
    even with the same seed unless DataLoader worker seeds are also fixed.
    GPU operations can introduce additional nondeterminism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SASRec sequential recommender on MovieLens-1M.")

    parser.add_argument("--data", type=str,
                        default=os.path.join(project_root, "data", "processed", "ml-1m.txt"),
                        help="Path to the processed interaction data file.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Number of users per training batch (default: 128).")
    parser.add_argument("--maxlen", type=int, default=50,
                        help="Sequence context length (default: 50).")
    parser.add_argument("--hidden-units", type=int, default=50,
                        help="Hidden dimensionality of the model (default: 50).")
    parser.add_argument("--num-blocks", type=int, default=2,
                        help="Number of self-attention blocks (default: 2).")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="Number of attention heads (default: 1).")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability (default: 0.5).")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Adam learning rate (default: 0.001).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on: 'cpu' or 'cuda'. Defaults to cuda if available.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit each epoch to this many batches. Omit for full training.")
    parser.add_argument("--checkpoint-path", type=str,
                        default=None,
                        help="Path to save the model checkpoint. Defaults to dynamic name based on configuration.")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    # ---- Resolve dynamic checkpoint path ----
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        if args.maxlen == 50 and args.dropout == 0.5:
            checkpoint_filename = "sasrec_movielens_baseline.pt"
        else:
            dropout_str = str(args.dropout).replace('.', '')
            checkpoint_filename = f"sasrec_movielens_maxlen{args.maxlen}_dropout{dropout_str}.pt"
        checkpoint_path = os.path.join(project_root, "results", "checkpoints", checkpoint_filename)

    # ---- Device selection ----
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("Error: --device cuda was requested but CUDA is not available.")
            sys.exit(1)

    is_dev_run = args.max_batches is not None

    # ---- Header ----
    print("==============================================")
    if is_dev_run:
        print("SASRec Training  [DEVELOPMENT / SANITY RUN]")
    else:
        print("SASRec Training  [FULL RUN]")
    print("==============================================\n")

    # ---- Load dataset ----
    if not os.path.exists(args.data):
        print(f"Error: Processed dataset not found at: {args.data}")
        sys.exit(1)

    print(f"Loading dataset from: {args.data}")
    dataset = load_dataset(args.data)
    user_train = dataset["train"]
    item_count = dataset["item_count"]
    print(f"  Users: {dataset['user_count']}  |  Items: {item_count}\n")

    # ---- Print configuration ----
    print("Training configuration")
    print("----------------------")
    print(f"  Dataset:      {args.data}")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Maxlen:       {args.maxlen}")
    print(f"  Hidden units: {args.hidden_units}")
    print(f"  Blocks:       {args.num_blocks}")
    print(f"  Heads:        {args.num_heads}")
    print(f"  Dropout:      {args.dropout}")
    print(f"  Learning rate:{args.lr}")
    print(f"  Seed:         {args.seed}")
    if is_dev_run:
        print(f"  Max batches:  {args.max_batches}  (dev run)")
    print(f"  Checkpoint:   {checkpoint_path}")
    print()

    # ---- DataLoader ----
    train_dataset = SASRecDataset(user_train, item_count, args.maxlen)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0        # keep simple; avoids multiprocess issues on Windows
    )
    total_batches = len(train_loader)
    print(f"Batches per epoch: {total_batches}")
    if is_dev_run:
        print(f"Running first {args.max_batches} batches per epoch (dev mode).\n")

    # ---- Model & Trainer ----
    model = SASRec(
        item_count=item_count,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout
    )
    trainer = SASRecTrainer(model=model, lr=args.lr, device=str(device))

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}\n")

    # ---- Training loop ----
    print("Starting training...\n")
    t_start = time.time()
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches_run = 0

        for batch_idx, batch in enumerate(train_loader):
            if is_dev_run and batch_idx >= args.max_batches:
                break

            loss = trainer.train_step(batch)
            epoch_loss += loss
            batches_run += 1

        avg_loss = epoch_loss / batches_run if batches_run > 0 else float("nan")
        loss_history.append(avg_loss)
        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{args.epochs}  |  Loss: {avg_loss:.4f}  |  Elapsed: {elapsed:.1f}s")

    # ---- Save checkpoint ----
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = {
        # Weights
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        # Training state
        "epoch": args.epochs,
        "loss_history": loss_history,
        # Architecture config (needed by evaluate.py to reconstruct the model)
        "item_count": item_count,
        "maxlen": args.maxlen,
        "hidden_units": args.hidden_units,
        "num_blocks": args.num_blocks,
        "num_heads": args.num_heads,
        "dropout_rate": args.dropout,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    print("==============================================")


if __name__ == "__main__":
    main()
