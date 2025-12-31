"""
Evaluation Metrics for Recommender Systems.

This module provides standard metrics for evaluating recommendation quality,
including ranking metrics (NDCG, MAP) and retrieval metrics (Hit Rate, Recall).

Professor SPARK's Note:
    "The right metric depends on your goal. Optimizing clicks? Use Hit Rate.
    Care about ranking? Use NDCG. Always know what you're optimizing for!"
"""

import numpy as np
from typing import List, Optional, Union, Dict
import torch


# =============================================================================
# Point-wise Metrics (for explicit ratings)
# =============================================================================

def rmse(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> float:
    """
    Root Mean Squared Error.

    Lower is better. Common for explicit rating prediction.

    Args:
        predictions: Predicted ratings
        actuals: Actual ratings

    Returns:
        RMSE value

    Example:
        >>> preds = np.array([3.5, 4.0, 2.5])
        >>> actual = np.array([4.0, 4.0, 3.0])
        >>> print(f"RMSE: {rmse(preds, actual):.3f}")
        RMSE: 0.408
    """
    return np.sqrt(np.mean((predictions - actuals) ** 2))


def mae(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> float:
    """
    Mean Absolute Error.

    More robust to outliers than RMSE.

    Args:
        predictions: Predicted ratings
        actuals: Actual ratings

    Returns:
        MAE value
    """
    return np.mean(np.abs(predictions - actuals))


# =============================================================================
# Ranking Metrics
# =============================================================================

def hit_rate_at_k(
    ranked_items: np.ndarray,
    ground_truth: Union[int, List[int]],
    k: int = 10
) -> float:
    """
    Hit Rate @ K: Was the relevant item in the top K predictions?

    This is the most common metric for implicit feedback evaluation.
    Think of it as: "Did we recommend at least one relevant item?"

    Args:
        ranked_items: Items sorted by predicted score (highest first)
        ground_truth: The item(s) the user actually interacted with
        k: Cutoff position

    Returns:
        1.0 if hit, 0.0 if miss

    Example:
        >>> predictions = [5, 2, 8, 1, 9]  # Item IDs, ranked by score
        >>> ground_truth = 8  # User actually liked item 8
        >>> hr = hit_rate_at_k(predictions, ground_truth, k=5)
        >>> print(f"Hit Rate @ 5: {hr}")  # 1.0 because item 8 is in top 5
        Hit Rate @ 5: 1.0
    """
    top_k = ranked_items[:k]

    if isinstance(ground_truth, (int, np.integer)):
        ground_truth = [ground_truth]

    for item in ground_truth:
        if item in top_k:
            return 1.0

    return 0.0


def recall_at_k(
    ranked_items: np.ndarray,
    ground_truth: List[int],
    k: int = 10
) -> float:
    """
    Recall @ K: What fraction of relevant items are in top K?

    Use when users have multiple relevant items.

    Args:
        ranked_items: Items sorted by predicted score
        ground_truth: List of relevant items
        k: Cutoff position

    Returns:
        Recall value between 0 and 1

    Example:
        >>> predictions = [5, 2, 8, 1, 9, 3, 4, 6, 7, 10]
        >>> ground_truth = [2, 8, 7]  # User liked 3 items
        >>> recall = recall_at_k(predictions, ground_truth, k=5)
        >>> print(f"Recall @ 5: {recall:.2f}")  # 2/3 = 0.67
        Recall @ 5: 0.67
    """
    if len(ground_truth) == 0:
        return 0.0

    top_k = set(ranked_items[:k])
    relevant = set(ground_truth)

    hits = len(top_k & relevant)
    return hits / len(relevant)


def precision_at_k(
    ranked_items: np.ndarray,
    ground_truth: List[int],
    k: int = 10
) -> float:
    """
    Precision @ K: What fraction of top K items are relevant?

    Args:
        ranked_items: Items sorted by predicted score
        ground_truth: List of relevant items
        k: Cutoff position

    Returns:
        Precision value between 0 and 1
    """
    top_k = set(ranked_items[:k])
    relevant = set(ground_truth)

    hits = len(top_k & relevant)
    return hits / k


def dcg_at_k(
    relevances: np.ndarray,
    k: int = 10
) -> float:
    """
    Discounted Cumulative Gain @ K.

    Measures ranking quality with position-based discounting.
    Items at the top matter more than items at the bottom.

    Args:
        relevances: Relevance scores in ranked order
        k: Cutoff position

    Returns:
        DCG value

    Example:
        >>> # Perfect ranking: [3, 2, 1] (most relevant first)
        >>> dcg = dcg_at_k(np.array([3, 2, 1]), k=3)
        >>> print(f"DCG: {dcg:.3f}")
        DCG: 4.631
    """
    relevances = np.array(relevances)[:k]

    # Discounts: log2(2), log2(3), log2(4), ...
    discounts = np.log2(np.arange(2, len(relevances) + 2))

    return np.sum(relevances / discounts)


def ndcg_at_k(
    ranked_items: np.ndarray,
    ground_truth: Union[List[int], Dict[int, float]],
    k: int = 10
) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    The gold standard for ranking evaluation. Rewards:
    1. Relevant items appearing in recommendations
    2. Relevant items appearing higher in the list

    Normalized by ideal DCG so result is between 0 and 1.

    Args:
        ranked_items: Items sorted by predicted score (highest first)
        ground_truth: List of relevant items, or dict of {item: relevance}
        k: Cutoff position

    Returns:
        NDCG value between 0 and 1

    Example:
        >>> # Our model's ranking
        >>> predictions = [5, 2, 8, 1, 9]
        >>> # Items 2 and 8 are relevant (binary relevance)
        >>> ground_truth = [2, 8]
        >>> ndcg = ndcg_at_k(predictions, ground_truth, k=5)
        >>> print(f"NDCG @ 5: {ndcg:.3f}")
        NDCG @ 5: 0.631
    """
    # Convert to relevance dict if list provided
    if isinstance(ground_truth, list):
        ground_truth = {item: 1.0 for item in ground_truth}

    # Get relevances for ranked items
    relevances = []
    for item in ranked_items[:k]:
        relevances.append(ground_truth.get(item, 0.0))

    relevances = np.array(relevances)

    # Compute DCG
    dcg = dcg_at_k(relevances, k)

    # Compute ideal DCG (perfect ranking)
    ideal_relevances = sorted(ground_truth.values(), reverse=True)[:k]

    # Pad if fewer relevant items than k
    if len(ideal_relevances) < k:
        ideal_relevances = list(ideal_relevances) + [0.0] * (k - len(ideal_relevances))

    idcg = dcg_at_k(np.array(ideal_relevances), k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision(
    ranked_items: np.ndarray,
    ground_truth: List[int],
    k: Optional[int] = None
) -> float:
    """
    Average Precision: Average of precision at each relevant position.

    Rewards both precision AND recall in a single metric.

    Args:
        ranked_items: Items sorted by predicted score
        ground_truth: List of relevant items
        k: Cutoff position (optional)

    Returns:
        AP value between 0 and 1
    """
    if len(ground_truth) == 0:
        return 0.0

    if k is not None:
        ranked_items = ranked_items[:k]

    relevant = set(ground_truth)
    hits = 0
    sum_precisions = 0.0

    for i, item in enumerate(ranked_items):
        if item in relevant:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i

    if hits == 0:
        return 0.0

    # Normalize by total relevant items (bounded by k if specified)
    normalizer = min(len(ground_truth), k) if k else len(ground_truth)
    return sum_precisions / normalizer


def mean_average_precision(
    all_ranked_items: List[np.ndarray],
    all_ground_truths: List[List[int]],
    k: int = 10
) -> float:
    """
    Mean Average Precision @ K across all users.

    Args:
        all_ranked_items: List of ranked items for each user
        all_ground_truths: List of relevant items for each user
        k: Cutoff position

    Returns:
        MAP value between 0 and 1

    Example:
        >>> # 3 users' predictions and ground truths
        >>> predictions = [
        ...     np.array([1, 2, 3, 4, 5]),  # User 1
        ...     np.array([6, 7, 8, 9, 10]),  # User 2
        ...     np.array([11, 12, 13, 14, 15])  # User 3
        ... ]
        >>> ground_truths = [
        ...     [2, 4],  # User 1 likes items 2, 4
        ...     [7],     # User 2 likes item 7
        ...     [15]     # User 3 likes item 15
        ... ]
        >>> map_score = mean_average_precision(predictions, ground_truths, k=5)
    """
    aps = []

    for ranked, gt in zip(all_ranked_items, all_ground_truths):
        ap = average_precision(ranked, gt, k)
        aps.append(ap)

    return np.mean(aps)


def mrr(
    ranked_items: np.ndarray,
    ground_truth: Union[int, List[int]]
) -> float:
    """
    Mean Reciprocal Rank: 1 / position of first relevant item.

    Use when you only care about the first relevant recommendation.

    Args:
        ranked_items: Items sorted by predicted score
        ground_truth: Relevant item(s)

    Returns:
        MRR value between 0 and 1

    Example:
        >>> predictions = [5, 2, 8, 1, 9]
        >>> ground_truth = 8  # Item 8 is at position 3
        >>> print(f"MRR: {mrr(predictions, ground_truth):.3f}")
        MRR: 0.333
    """
    if isinstance(ground_truth, (int, np.integer)):
        ground_truth = [ground_truth]

    relevant = set(ground_truth)

    for i, item in enumerate(ranked_items):
        if item in relevant:
            return 1.0 / (i + 1)

    return 0.0


# =============================================================================
# Aggregate Metrics
# =============================================================================

def evaluate_ranking(
    all_ranked_items: List[np.ndarray],
    all_ground_truths: List[List[int]],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Comprehensive ranking evaluation with multiple metrics.

    Args:
        all_ranked_items: Ranked item lists for each user
        all_ground_truths: Ground truth items for each user
        k_values: K values to evaluate

    Returns:
        Dictionary of metric_name -> value

    Example:
        >>> metrics = evaluate_ranking(predictions, ground_truths, k_values=[5, 10])
        >>> print(f"NDCG@10: {metrics['ndcg@10']:.3f}")
        >>> print(f"HR@10: {metrics['hr@10']:.3f}")
    """
    results = {}

    for k in k_values:
        # Hit Rate
        hr_scores = [
            hit_rate_at_k(ranked, gt, k)
            for ranked, gt in zip(all_ranked_items, all_ground_truths)
        ]
        results[f'hr@{k}'] = np.mean(hr_scores)

        # NDCG
        ndcg_scores = [
            ndcg_at_k(ranked, gt, k)
            for ranked, gt in zip(all_ranked_items, all_ground_truths)
        ]
        results[f'ndcg@{k}'] = np.mean(ndcg_scores)

        # Recall
        recall_scores = [
            recall_at_k(ranked, gt, k)
            for ranked, gt in zip(all_ranked_items, all_ground_truths)
        ]
        results[f'recall@{k}'] = np.mean(recall_scores)

    # MRR (K-independent)
    mrr_scores = [
        mrr(ranked, gt)
        for ranked, gt in zip(all_ranked_items, all_ground_truths)
    ]
    results['mrr'] = np.mean(mrr_scores)

    # MAP
    results[f'map@{max(k_values)}'] = mean_average_precision(
        all_ranked_items, all_ground_truths, k=max(k_values)
    )

    return results


# =============================================================================
# Coverage and Diversity Metrics
# =============================================================================

def coverage(
    all_recommendations: List[List[int]],
    total_items: int
) -> float:
    """
    Catalog Coverage: Fraction of items ever recommended.

    Low coverage suggests popularity bias.

    Args:
        all_recommendations: All recommendations made
        total_items: Total number of items in catalog

    Returns:
        Coverage value between 0 and 1
    """
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)

    return len(recommended_items) / total_items


def intra_list_diversity(
    recommendations: List[int],
    similarity_matrix: np.ndarray
) -> float:
    """
    Intra-List Diversity: Average dissimilarity between recommended items.

    Higher diversity prevents "echo chambers."

    Args:
        recommendations: List of recommended item IDs
        similarity_matrix: Pre-computed item-item similarity matrix

    Returns:
        ILD value between 0 and 1
    """
    if len(recommendations) < 2:
        return 0.0

    total_dissim = 0.0
    count = 0

    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            item_i = recommendations[i]
            item_j = recommendations[j]
            dissim = 1 - similarity_matrix[item_i, item_j]
            total_dissim += dissim
            count += 1

    return total_dissim / count


def novelty(
    recommendations: List[int],
    item_popularity: np.ndarray
) -> float:
    """
    Novelty: Average inverse popularity of recommendations.

    Rewards recommending less popular (surprising) items.

    Args:
        recommendations: Recommended item IDs
        item_popularity: Array of popularity scores per item

    Returns:
        Novelty score
    """
    if len(recommendations) == 0:
        return 0.0

    # Avoid log(0)
    pop = item_popularity[recommendations]
    pop = np.maximum(pop, 1e-10)

    # Self-information: -log2(popularity)
    self_info = -np.log2(pop / pop.sum())

    return np.mean(self_info)


# =============================================================================
# Utility Functions
# =============================================================================

def format_metrics(
    metrics: Dict[str, float],
    title: str = "Evaluation Results"
) -> str:
    """
    Format metrics dictionary as a nice table.

    Example:
        >>> print(format_metrics({'ndcg@10': 0.542, 'hr@10': 0.721}))
        ════════════════════════════════
        Evaluation Results
        ════════════════════════════════
        NDCG@10        0.5420
        HR@10          0.7210
        ════════════════════════════════
    """
    lines = ["═" * 35]
    lines.append(title)
    lines.append("═" * 35)

    for metric, value in sorted(metrics.items()):
        # Capitalize and format metric name
        name = metric.upper().replace("_", " ")
        lines.append(f"{name:<15} {value:.4f}")

    lines.append("═" * 35)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")

    # Test Hit Rate
    ranked = np.array([5, 2, 8, 1, 9, 3, 7, 4, 6, 10])
    gt = [8]
    hr = hit_rate_at_k(ranked, gt, k=5)
    assert hr == 1.0, f"Expected 1.0, got {hr}"
    print(f"✅ Hit Rate @ 5: {hr}")

    # Test NDCG
    ndcg = ndcg_at_k(ranked, gt, k=10)
    print(f"✅ NDCG @ 10: {ndcg:.4f}")

    # Test with multiple relevant items
    gt_multi = [2, 8, 7]
    recall = recall_at_k(ranked, gt_multi, k=5)
    print(f"✅ Recall @ 5: {recall:.4f} (2 of 3 items found)")

    # Test RMSE
    preds = np.array([3.5, 4.0, 2.5, 4.5, 3.0])
    actual = np.array([4.0, 4.0, 3.0, 5.0, 2.5])
    rmse_val = rmse(preds, actual)
    print(f"✅ RMSE: {rmse_val:.4f}")

    # Test comprehensive evaluation
    all_ranked = [
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        np.array([5, 6, 7, 8, 9, 10, 1, 2, 3, 4])
    ]
    all_gt = [[2, 5], [7, 10]]

    metrics = evaluate_ranking(all_ranked, all_gt, k_values=[5, 10])
    print("\n" + format_metrics(metrics))

    print("\n✅ All metric tests passed!")
