from collections import Counter, defaultdict
from pathlib import Path


def preprocess_movielens(input_path: str, output_path: str) -> None:
    user_counts = Counter()
    item_counts = Counter()
    interactions = []

    # Read raw MovieLens ratings.
    with open(input_path, "r", encoding="latin-1") as f:
        for line in f:
            user_id, item_id, _rating, timestamp = line.strip().split("::")

            user_id = int(user_id)
            item_id = int(item_id)
            timestamp = int(timestamp)

            interactions.append((user_id, item_id, timestamp))
            user_counts[user_id] += 1
            item_counts[item_id] += 1

    # Keep users and items with at least 5 interactions.
    filtered = [
        (user_id, item_id, timestamp)
        for user_id, item_id, timestamp in interactions
        if user_counts[user_id] >= 5
        and item_counts[item_id] >= 5
    ]

    # Remap IDs to contiguous values starting from 1.
    user_map = {}
    item_map = {}
    user_sequences = defaultdict(list)

    next_user_id = 1
    next_item_id = 1

    for user_id, item_id, timestamp in filtered:
        if user_id not in user_map:
            user_map[user_id] = next_user_id
            next_user_id += 1

        if item_id not in item_map:
            item_map[item_id] = next_item_id
            next_item_id += 1

        new_user_id = user_map[user_id]
        new_item_id = item_map[item_id]

        user_sequences[new_user_id].append(
            (timestamp, new_item_id)
        )

    # Sort each user's interactions chronologically.
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[0])

    # Save as:
    # user_id item_id
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for user_id in sorted(user_sequences):
            for _, item_id in user_sequences[user_id]:
                f.write(f"{user_id} {item_id}\n")

    print(f"Users: {len(user_sequences)}")
    print(f"Items: {len(item_map)}")
    print(f"Interactions: {len(filtered)}")


if __name__ == "__main__":
    preprocess_movielens(
        "data/raw/ml-1m/ratings.dat",
        "data/processed/ml-1m.txt",
    )