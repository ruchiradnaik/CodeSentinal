def normalize_scores(scores):
    total = sum(scores)
    
    # Check if total is zero to avoid division by zero
    if total == 0:
        return [0] * len(scores)  # Return a list of zeros if total is zero

    normalized = []
    for s in scores:
        normalized.append(s / total)

    return normalized

data = [10, 20, 30, 40]
result = normalize_scores(data)
print(result)