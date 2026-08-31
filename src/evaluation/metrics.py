import math


def hit_rate_at_k(rank: int, k: int = 10) -> float:
    """
    Computes Hit Rate at K (HR@K).

    Args:
        rank: The 0-based rank of the true item.
        k: The threshold for top-K recommendation.

    Returns:
        1.0 if the true item is ranked in the top K (rank < k), else 0.0.
    """
    return 1.0 if rank < k else 0.0


def ndcg_at_k(rank: int, k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (NDCG@K).

    For sequential recommendations with a single true item, this evaluates
    to 1 / log2(rank + 2) if the rank is within top-K, else 0.0.

    Args:
        rank: The 0-based rank of the true item.
        k: The threshold for top-K recommendation.

    Returns:
        NDCG score float.
    """
    if rank < k:
        return 1.0 / math.log2(rank + 2)
    return 0.0


if __name__ == "__main__":
    print("Running metrics sanity tests...")
    
    # Test case 1: ranked first (0-based rank 0)
    assert hit_rate_at_k(0, 10) == 1.0
    assert ndcg_at_k(0, 10) == 1.0
    print("  - Rank 0 (1st place) passed.")

    # Test case 2: ranked tenth (0-based rank 9)
    assert hit_rate_at_k(9, 10) == 1.0
    assert abs(ndcg_at_k(9, 10) - (1.0 / math.log2(11))) < 1e-9
    print("  - Rank 9 (10th place) passed.")

    # Test case 3: ranked eleventh (0-based rank 10)
    assert hit_rate_at_k(10, 10) == 0.0
    assert ndcg_at_k(10, 10) == 0.0
    print("  - Rank 10 (11th place) passed.")

    # Test case 4: ranked outside (0-based rank 15)
    assert hit_rate_at_k(15, 10) == 0.0
    assert ndcg_at_k(15, 10) == 0.0
    print("  - Rank 15 passed.")

    print("ALL METRICS SANITY TESTS PASSED!")
