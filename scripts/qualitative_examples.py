import os
import sys
import torch
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import load_dataset
from src.models.sasrec import SASRec
from src.evaluation.evaluator import build_eval_sequence

def load_item_mapping(ratings_path: str):
    from collections import Counter
    user_counts = Counter()
    item_counts = Counter()
    interactions = []

    with open(ratings_path, "r", encoding="latin-1") as f:
        for line in f:
            user_id, item_id, _rating, timestamp = line.strip().split("::")
            user_id = int(user_id)
            item_id = int(item_id)
            timestamp = int(timestamp)
            interactions.append((user_id, item_id, timestamp))
            user_counts[user_id] += 1
            item_counts[item_id] += 1

    filtered = [
        (user_id, item_id, timestamp)
        for user_id, item_id, timestamp in interactions
        if user_counts[user_id] >= 5
        and item_counts[item_id] >= 5
    ]

    item_map = {}
    next_item_id = 1
    for _, item_id, _ in filtered:
        if item_id not in item_map:
            item_map[item_id] = next_item_id
            next_item_id += 1

    return item_map

def load_movies_metadata(movies_path: str):
    movie_titles = {}
    movie_genres = {}
    with open(movies_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 2:
                movie_id = int(parts[0])
                title = parts[1]
                genres = parts[2] if len(parts) > 2 else ""
                movie_titles[movie_id] = title
                movie_genres[movie_id] = genres
    return movie_titles, movie_genres

def main():
    print("Loading datasets and mappings...")
    ratings_path = os.path.join(project_root, "data", "raw", "ml-1m", "ratings.dat")
    movies_path = os.path.join(project_root, "data", "raw", "ml-1m", "movies.dat")
    processed_path = os.path.join(project_root, "data", "processed", "ml-1m.txt")

    item_map = load_item_mapping(ratings_path)
    reverse_item_map = {v: k for k, v in item_map.items()}
    movie_titles, movie_genres = load_movies_metadata(movies_path)

    dataset = load_dataset(processed_path)
    train_seqs = dataset["train"]
    valid_seqs = dataset["valid"]
    test_seqs = dataset["test"]

    checkpoint_path = os.path.join(project_root, "results", "checkpoints", "sasrec_movielens_maxlen200_dropout02.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SASRec(
        item_count=checkpoint["item_count"],
        maxlen=checkpoint["maxlen"],
        hidden_units=checkpoint["hidden_units"],
        num_blocks=checkpoint["num_blocks"],
        num_heads=checkpoint["num_heads"],
        dropout_rate=0.0
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    def get_movie_str(mapped_id):
        orig_id = reverse_item_map.get(mapped_id)
        if orig_id is None:
            return f"Unknown (Mapped ID: {mapped_id})"
        title = movie_titles.get(orig_id, "Unknown Title")
        genres = movie_genres.get(orig_id, "Unknown Genres")
        return f"{title} [{genres}] (ID: {orig_id})"

    # Select deterministic users representing different behavior/sizes
    target_users = [1, 10, 50, 100, 500]

    for u in target_users:
        history = train_seqs.get(u, [])
        valid_item = valid_seqs.get(u, [None])[0]
        test_item = test_seqs.get(u, [None])[0]

        if not history:
            continue

        # Build sequence and make prediction
        seq = build_eval_sequence(history, checkpoint["maxlen"])
        seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.no_grad():
            seq_rep = model(seq_tensor)
            final_rep = seq_rep[0, -1]
            all_emb = model.item_emb.weight
            scores = torch.matmul(all_emb, final_rep).numpy()

        # Exclude history and padding index 0
        for item in history:
            scores[item] = -1e9
        scores[0] = -1e9

        top5_indices = np.argsort(scores)[-5:][::-1]

        print(f"\n==========================================")
        print(f"User ID: {u}")
        print(f"==========================================")
        print("Recent Watch History (last 10 items):")
        for item in history[-10:]:
            print(f"  - {get_movie_str(item)}")
        
        print("\nHeld-out targets:")
        print(f"  Validation (Next): {get_movie_str(valid_item) if valid_item else 'None'}")
        print(f"  Test (After Valid): {get_movie_str(test_item) if test_item else 'None'}")

        print("\nTop-5 Recommended Movies:")
        for idx, rec_id in enumerate(top5_indices):
            is_valid_target = (rec_id == valid_item)
            is_test_target = (rec_id == test_item)
            marker = ""
            if is_valid_target:
                marker = " [VAL MATCH]"
            elif is_test_target:
                marker = " [TEST MATCH]"
            print(f"  {idx+1}. {get_movie_str(rec_id)}{marker}")

if __name__ == "__main__":
    main()
