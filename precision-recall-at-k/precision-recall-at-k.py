def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    relevant_set = set(relevant)
    recommended_k = recommended[0:k]
    hits = 0
    for element in recommended_k:
        if element in relevant_set:
            hits += 1
    Precision_k = hits/k
    recall_k = hits/len(relevant)
    
    return [Precision_k, recall_k]