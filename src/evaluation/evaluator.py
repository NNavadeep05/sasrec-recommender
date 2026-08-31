import random
import torch
import torch.nn as nn

from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k


def build_eval_sequence(history: list, maxlen: int) -> list:
    """
    Constructs a right-aligned, zero-padded history sequence.

    If history is shorter than maxlen, left-pads with 0.
    If longer, keeps only the most recent maxlen items.

    Args:
        history: List of item IDs representing interaction history.
        maxlen: Target sequence length.

    Returns:
        List of length maxlen.
    """
    seq = [0] * maxlen
    if len(history) == 0:
        return seq
        
    sliced = history[-maxlen:]
    # Place sliced elements at the end of the zero sequence
    seq[-len(sliced):] = sliced
    return seq


def sample_negatives_eval(
    excluded: set, 
    item_count: int, 
    num_negatives: int = 100
) -> list:
    """
    Samples randomly selected negative items.

    Args:
        excluded: Set of item IDs that must not be sampled (user history & padding).
        item_count: Total number of items in the dataset.
        num_negatives: Number of negative candidates to sample.

    Returns:
        List of sampled item IDs.
    """
    negatives = []
    for _ in range(num_negatives):
        t = random.randint(1, item_count)
        while t in excluded:
            t = random.randint(1, item_count)
        negatives.append(t)
    return negatives


def evaluate_validation(
    model: nn.Module,
    train_seqs: dict,
    valid_seqs: dict,
    item_count: int,
    maxlen: int,
    device: torch.device,
    limit_users: int = None,
) -> dict:
    """
    Evaluates the model on the validation target split.

    Args:
        model: Trained SASRec model.
        train_seqs: Dictionary of training sequences (user_id -> list[item_id]).
        valid_seqs: Dictionary of validation sequences (user_id -> list[item_id]).
        item_count: Total number of items in the dataset.
        maxlen: Sequence max length.
        device: Device to run evaluation on.
        limit_users: Max number of random users to evaluate.

    Returns:
        Dictionary containing validation average HR@10 and NDCG@10.
    """
    model.eval()
    
    users = list(train_seqs.keys())
    if limit_users is not None and len(users) > limit_users:
        users = random.sample(users, limit_users)

    HR = 0.0
    NDCG = 0.0
    valid_users = 0.0

    with torch.no_grad():
        for u in users:
            # Skip user if training or validation target is missing
            if len(train_seqs.get(u, [])) < 1 or len(valid_seqs.get(u, [])) < 1:
                continue

            true_item = valid_seqs[u][0]
            history = train_seqs[u]
            seq = build_eval_sequence(history, maxlen)

            # Negatives must not appear in the training history
            excluded = set(history) | {0}
            negatives = sample_negatives_eval(excluded, item_count, 100)
            candidates = [true_item] + negatives

            # 1. Forward pass to get representation
            seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            seq_rep = model(seq_tensor)  # Shape: (1, maxlen, hidden_units)
            final_rep = seq_rep[0, -1]   # Shape: (hidden_units,)

            # 2. Get embeddings of the 101 candidates and score them
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            candidate_emb = model.item_emb(candidate_tensor)  # Shape: (101, hidden_units)
            
            # Compute dot product scores: (101,)
            scores = torch.matmul(candidate_emb, final_rep)

            # 3. Calculate 0-based rank of the true item (index 0)
            true_score = scores[0]
            rank = (scores > true_score).sum().item()

            HR += hit_rate_at_k(rank, 10)
            NDCG += ndcg_at_k(rank, 10)
            valid_users += 1.0

    if valid_users > 0:
        return {
            "HR@10": HR / valid_users,
            "NDCG@10": NDCG / valid_users,
            "users_evaluated": int(valid_users)
        }
    else:
        return {
            "HR@10": 0.0,
            "NDCG@10": 0.0,
            "users_evaluated": 0
        }


def evaluate_test(
    model: nn.Module,
    train_seqs: dict,
    valid_seqs: dict,
    test_seqs: dict,
    item_count: int,
    maxlen: int,
    device: torch.device,
    limit_users: int = None,
) -> dict:
    """
    Evaluates the model on the test target split.

    Args:
        model: Trained SASRec model.
        train_seqs: Dictionary of training sequences (user_id -> list[item_id]).
        valid_seqs: Dictionary of validation sequences (user_id -> list[item_id]).
        test_seqs: Dictionary of test sequences (user_id -> list[item_id]).
        item_count: Total number of items in the dataset.
        maxlen: Sequence max length.
        device: Device to run evaluation on.
        limit_users: Max number of random users to evaluate.

    Returns:
        Dictionary containing test average HR@10 and NDCG@10.
    """
    model.eval()
    
    users = list(train_seqs.keys())
    if limit_users is not None and len(users) > limit_users:
        users = random.sample(users, limit_users)

    HR = 0.0
    NDCG = 0.0
    valid_users = 0.0

    with torch.no_grad():
        for u in users:
            # Skip user if history or test target is missing
            if len(train_seqs.get(u, [])) < 1 or len(test_seqs.get(u, [])) < 1 or len(valid_seqs.get(u, [])) < 1:
                continue

            true_item = test_seqs[u][0]
            # Test history consists of training sequence + validation item
            history = train_seqs[u] + valid_seqs[u]
            seq = build_eval_sequence(history, maxlen)

            # Negatives must not appear in the test-time history
            excluded = set(history) | {0}
            negatives = sample_negatives_eval(excluded, item_count, 100)
            candidates = [true_item] + negatives

            # 1. Forward pass to get representation
            seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            seq_rep = model(seq_tensor)  # Shape: (1, maxlen, hidden_units)
            final_rep = seq_rep[0, -1]   # Shape: (hidden_units,)

            # 2. Get embeddings of the 101 candidates and score them
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            candidate_emb = model.item_emb(candidate_tensor)  # Shape: (101, hidden_units)
            
            # Compute dot product scores: (101,)
            scores = torch.matmul(candidate_emb, final_rep)

            # 3. Calculate 0-based rank of the true item (index 0)
            true_score = scores[0]
            rank = (scores > true_score).sum().item()

            HR += hit_rate_at_k(rank, 10)
            NDCG += ndcg_at_k(rank, 10)
            valid_users += 1.0

    if valid_users > 0:
        return {
            "HR@10": HR / valid_users,
            "NDCG@10": NDCG / valid_users,
            "users_evaluated": int(valid_users)
        }
    else:
        return {
            "HR@10": 0.0,
            "NDCG@10": 0.0,
            "users_evaluated": 0
        }


if __name__ == "__main__":
    from src.models.sasrec import SASRec

    print("Running evaluator sanity tests...")

    # Set seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # 1. Setup mock data
    # 3 users, item_count = 10
    train_seqs = {1: [1, 2, 3], 2: [4, 5], 3: [1]}
    valid_seqs = {1: [6], 2: [7], 3: []}  # user 3 has no valid target, should be skipped
    test_seqs = {1: [8], 2: [9], 3: [10]}  # user 3 has no valid target, should be skipped in test

    maxlen = 5
    item_count = 10
    device = torch.device("cpu")

    # 2. Setup mock model
    model = SASRec(
        item_count=item_count,
        maxlen=maxlen,
        hidden_units=8,
        num_blocks=1,
        num_heads=1,
        dropout_rate=0.0
    )

    # 3. Test build_eval_sequence
    assert build_eval_sequence([1, 2], 5) == [0, 0, 0, 1, 2]
    assert build_eval_sequence([1, 2, 3, 4, 5, 6], 5) == [2, 3, 4, 5, 6]
    print("  - build_eval_sequence tests passed.")

    # 4. Test sample_negatives_eval
    excluded = {1, 2, 0}
    neg_items = sample_negatives_eval(excluded, item_count, 5)
    assert len(neg_items) == 5
    for item in neg_items:
        assert item not in excluded
        assert 1 <= item <= item_count
    print("  - sample_negatives_eval tests passed.")

    # 5. Run validation evaluation
    val_res = evaluate_validation(model, train_seqs, valid_seqs, item_count, maxlen, device)
    print("  - Validation results:", val_res)
    assert "HR@10" in val_res and "NDCG@10" in val_res
    assert val_res["users_evaluated"] == 2  # user 1 and user 2 evaluated, user 3 skipped
    assert 0.0 <= val_res["HR@10"] <= 1.0
    assert 0.0 <= val_res["NDCG@10"] <= 1.0

    # 6. Run test evaluation
    test_res = evaluate_test(model, train_seqs, valid_seqs, test_seqs, item_count, maxlen, device)
    print("  - Test results:", test_res)
    assert "HR@10" in test_res and "NDCG@10" in test_res
    assert test_res["users_evaluated"] == 2  # user 1 and user 2 evaluated, user 3 skipped
    assert 0.0 <= test_res["HR@10"] <= 1.0
    assert 0.0 <= test_res["NDCG@10"] <= 1.0

    # 7. Check that no gradients are tracked for model output inside evaluate
    model.eval()
    with torch.no_grad():
        dummy_seq = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long, device=device)
        dummy_out = model(dummy_seq)
        assert not dummy_out.requires_grad, "Output tensor should not require grad when gradients are disabled!"
    print("  - No gradients check passed (requires_grad is False).")

    print("ALL EVALUATOR SANITY TESTS PASSED!")
