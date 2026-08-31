import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path to guarantee import safety from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import load_dataset
from src.models.sasrec import SASRec
from src.training.trainer import SASRecDataset, SASRecTrainer


def main():
    print("====================================================")
    print("Starting SASRec End-to-End Real-Data Sanity Training")
    print("====================================================\n")

    # 1. Path setup
    dataset_path = os.path.join(project_root, "data", "processed", "ml-1m.txt")
    if not os.path.exists(dataset_path):
        print(f"Error: Processed dataset not found at {dataset_path}")
        sys.exit(1)

    # 2. Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    user_train = dataset["train"]
    user_count = dataset["user_count"]
    item_count = dataset["item_count"]
    
    print(f"Dataset Loaded Successfully:")
    print(f"  - Users count (in train): {len(user_train)}")
    print(f"  - Total User IDs: {user_count}")
    print(f"  - Total Item IDs: {item_count}")
    print(f"  - Total Interactions loaded: {len(user_train)}")
    print("-" * 52)

    # 3. Setup configurations
    maxlen = 50
    hidden_units = 50
    num_blocks = 2
    num_heads = 1
    dropout_rate = 0.5
    learning_rate = 0.001
    batch_size = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Configuration:")
    print(f"  - maxlen: {maxlen}")
    print(f"  - hidden_units: {hidden_units}")
    print(f"  - num_blocks: {num_blocks}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - dropout_rate: {dropout_rate}")
    print(f"  - learning_rate: {learning_rate}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - device: {device}")
    print("-" * 52)

    # 4. Instantiate PyTorch Dataset and DataLoader
    # SASRecDataset dynamically calls create_training_sample on-the-fly, keeping memory usage constant.
    train_dataset = SASRecDataset(user_train, item_count, maxlen)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    print(f"Data Loader Setup:")
    print(f"  - Total Batches per Epoch: {len(train_loader)}")
    print("-" * 52)

    # 5. Instantiate Model and Trainer
    model = SASRec(
        item_count=item_count,
        maxlen=maxlen,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate
    )
    
    trainer = SASRecTrainer(model=model, lr=learning_rate, device=device)
    print("Model and Trainer successfully instantiated.")
    print("-" * 52)

    # 6. Sanity check parameters before optimization
    initial_weights = model.item_emb.weight.clone().detach()

    # 7. Run a short sanity training loop
    print("Running short training sanity check (first 20 batches)...")
    model.train()
    
    total_loss = 0.0
    num_batches_to_run = 20
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches_to_run:
            break
            
        loss = trainer.train_step(batch)
        total_loss += loss
        
        # Verify gradients are produced
        has_grad = model.item_emb.weight.grad is not None
        grad_status = "Grad: Yes" if has_grad else "Grad: No"
        
        print(f"Batch {batch_idx + 1}/{num_batches_to_run} - Loss: {loss:.4f} | {grad_status}")

    # 8. Post-training checks
    avg_loss = total_loss / num_batches_to_run
    print("-" * 52)
    print("Sanity Run Complete:")
    print(f"  - Average Loss: {avg_loss:.4f}")
    
    # Check weight updates
    final_weights = model.item_emb.weight.clone().detach()
    weights_changed = not torch.allclose(initial_weights, final_weights)
    print(f"  - Weights updated by optimizer: {weights_changed}")
    
    # Check for NaN / Inf in model parameters
    nan_occurred = torch.isnan(model.item_emb.weight).any().item()
    print(f"  - NaN values in parameters: {nan_occurred}")

    # 9. Verify if weights changed & loss is valid
    if weights_changed and not nan_occurred and avg_loss > 0:
         print("\n>>> PIPELINE INTEGRATION SANITY CHECK: PASSED! <<<")
    else:
         print("\n>>> PIPELINE INTEGRATION SANITY CHECK: FAILED! <<<")
    print("====================================================")


if __name__ == "__main__":
    main()
