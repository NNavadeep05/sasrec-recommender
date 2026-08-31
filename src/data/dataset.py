from collections import defaultdict
from pathlib import Path


def load_sequences(path: str):
    """
    Load the processed SASRec-format file:

        user_id item_id

    Returns:
        user_sequences: dict[user_id, list[item_id]]
        user_count: number of users
        item_count: largest item ID
    """
    user_sequences = defaultdict(list)
    item_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            user_id, item_id = map(int, line.strip().split())

            user_sequences[user_id].append(item_id)
            item_count = max(item_count, item_id)

    return dict(user_sequences), len(user_sequences), item_count


def train_valid_test_split(user_sequences):
    """
    For each user:

        last item       -> test
        second-last     -> validation
        everything else -> training

    Users with fewer than 3 interactions do not get
    validation/test targets.
    """
    user_train = {}
    user_valid = {}
    user_test = {}

    for user_id, sequence in user_sequences.items():
        if len(sequence) < 3:
            user_train[user_id] = sequence
            user_valid[user_id] = []
            user_test[user_id] = []
            continue

        user_train[user_id] = sequence[:-2]
        user_valid[user_id] = [sequence[-2]]
        user_test[user_id] = [sequence[-1]]

    return user_train, user_valid, user_test


def load_dataset(path: str):
    """
    Load processed interactions and create train/validation/test splits.
    """
    path = Path(path)

    sequences, user_count, item_count = load_sequences(str(path))

    train, valid, test = train_valid_test_split(sequences)

    return {
        "train": train,
        "valid": valid,
        "test": test,
        "user_count": user_count,
        "item_count": item_count,
    }


if __name__ == "__main__":
    dataset = load_dataset("data/processed/ml-1m.txt")

    print("Users:", dataset["user_count"])
    print("Items:", dataset["item_count"])

    first_user = min(dataset["train"])

    print("Example user:", first_user)
    print("Train:", dataset["train"][first_user])
    print("Valid:", dataset["valid"][first_user])
    print("Test:", dataset["test"][first_user])