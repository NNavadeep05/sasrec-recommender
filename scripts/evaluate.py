import os
import sys
import argparse
import json
import torch

# Add project root to sys.path to guarantee import safety from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import load_dataset
from src.models.sasrec import SASRec
from src.evaluation.evaluator import evaluate_validation, evaluate_test


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SASRec checkpoint.")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the saved PyTorch model checkpoint (.pt or .pth file)."
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default=os.path.join(project_root, "data", "processed", "ml-1m.txt"), 
        help="Path to the processed interaction data file."
    )
    parser.add_argument(
        "--maxlen", 
        type=int, 
        default=None, 
        help="Context sequence length (overrides checkpoint config if provided)."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Target device for evaluation (e.g. 'cpu', 'cuda')."
    )
    parser.add_argument(
        "--limit-users", 
        type=int, 
        default=None, 
        help="Limit evaluation to a random subset of N users (useful for quick sanity runs)."
    )
    args = parser.parse_args()

    print("==============================================")
    print("SASRec Command-Line Evaluation Utility")
    print("==============================================\n")

    # 1. Verify files exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"Error: Processed data file not found at: {args.data}")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Using Device: {device}")
    print(f"Loading checkpoint from: {args.checkpoint}")

    # 2. Load Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 3. Retrieve model configurations
    item_count = checkpoint.get("item_count")
    maxlen = args.maxlen if args.maxlen is not None else checkpoint.get("maxlen", 50)
    hidden_units = checkpoint.get("hidden_units", 50)
    num_blocks = checkpoint.get("num_blocks", 2)
    num_heads = checkpoint.get("num_heads", 1)
    dropout_rate = checkpoint.get("dropout_rate", 0.5)

    print(f"Checkpoint Configurations Loaded:")
    print(f"  - Item Count: {item_count}")
    print(f"  - Maxlen: {maxlen}")
    print(f"  - Hidden Units: {hidden_units}")
    print(f"  - Number of Blocks: {num_blocks}")
    print(f"  - Number of Heads: {num_heads}")
    print(f"  - Dropout Rate: {dropout_rate}")
    print("-" * 46)

    # 4. Reconstruct model
    model = SASRec(
        item_count=item_count,
        maxlen=maxlen,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate
    )
    
    # 5. Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("Model reconstructed and weights loaded successfully.")
    print("-" * 46)

    # 6. Load processed data
    print(f"Loading dataset from: {args.data}")
    dataset = load_dataset(args.data)
    train_seqs = dataset["train"]
    valid_seqs = dataset["valid"]
    test_seqs = dataset["test"]
    print("Dataset loaded successfully.")
    print("-" * 46)

    # 7. Run evaluation
    print(f"Running validation evaluation (Limit users: {args.limit_users})...")
    val_results = evaluate_validation(
        model=model,
        train_seqs=train_seqs,
        valid_seqs=valid_seqs,
        item_count=item_count,
        maxlen=maxlen,
        device=device,
        limit_users=args.limit_users
    )

    print(f"Running test evaluation (Limit users: {args.limit_users})...")
    test_results = evaluate_test(
        model=model,
        train_seqs=train_seqs,
        valid_seqs=valid_seqs,
        test_seqs=test_seqs,
        item_count=item_count,
        maxlen=maxlen,
        device=device,
        limit_users=args.limit_users
    )

    print("\n" + "=" * 46)
    print("Evaluation Results")
    print("=" * 46)
    print("Validation:")
    print(f"  HR@10:   {val_results['HR@10']:.4f}")
    print(f"  NDCG@10: {val_results['NDCG@10']:.4f}")
    print(f"  Users:   {val_results['users_evaluated']}")
    print("-" * 46)
    print("Test:")
    print(f"  HR@10:   {test_results['HR@10']:.4f}")
    print(f"  NDCG@10: {test_results['NDCG@10']:.4f}")
    print(f"  Users:   {test_results['users_evaluated']}")
    print("=" * 46)

    # 8. Save results to JSON file
    results = {
        "validation": val_results,
        "test": test_results
    }
    
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "evaluation.json")
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation results to: {results_path}")
    print("==============================================")


if __name__ == "__main__":
    main()
