import random


def sample_negative(item_count: int, user_items: set[int]) -> int:
    """Sample an item the user has not interacted with."""
    while True:
        item = random.randint(1, item_count)
        if item not in user_items:
            return item


def create_training_sample(
    user_id: int,
    user_sequence: list[int],
    item_count: int,
    maxlen: int,
) -> dict:
    """
    Create one SASRec training sample.

    The sequence is right-aligned and left-padded with 0.

    For example:

        user_sequence = [A, B, C, D]

    produces:

        seq = [0, A, B, C]
        pos = [0, B, C, D]
        neg = [0, X, Y, Z]

    where X, Y, Z are negative items.
    """
    seq = [0] * maxlen
    pos = [0] * maxlen
    neg = [0] * maxlen

    if len(user_sequence) < 2:
        return {
            "user": user_id,
            "sequence": seq,
            "positive": pos,
            "negative": neg,
        }

    user_items = set(user_sequence)

    # Only the last maxlen + 1 items can contribute to the sample.
    sequence = user_sequence[-(maxlen + 1):]

    next_item = sequence[-1]
    index = maxlen - 1

    for item in reversed(sequence[:-1]):
        seq[index] = item
        pos[index] = next_item
        neg[index] = sample_negative(item_count, user_items)

        next_item = item
        index -= 1

        if index < 0:
            break

    return {
        "user": user_id,
        "sequence": seq,
        "positive": pos,
        "negative": neg,
    }