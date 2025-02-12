import numpy as np

def load_predictions(chat_log_path):
    """
    Parses `chat_tail.txt` and extracts the full ranked list of model predictions.

    Returns:
    - `predictions`: List of ranked lists of candidate answers.
    """
    predictions = []

    with open(chat_log_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("Answers: "):
            ranked_list = line.strip().split("Answers: ")[1].strip().split(" | ")  # Now correctly extracting candidates
            predictions.append(ranked_list)

    return predictions

def evaluate_link_prediction_from_log(chat_log_path, test_data, k_list=[1, 3, 10]):
    """
    Evaluates link prediction using stored ranked lists from `chat_tail.txt`.

    Metrics:
    - Hits@K
    - Mean Rank (MR)
    - Mean Reciprocal Rank (MRR)

    :param chat_log_path: Path to `chat_tail.txt`
    :param test_data: List of ground-truth triplets
    :param k_list: List of `K` values for Hits@K
    """
    predictions = load_predictions(chat_log_path)

    ranks = []
    hits_at_k = {k: 0 for k in k_list}

    for idx, sample in enumerate(test_data):
        question = sample["Question"]
        true_answer = sample["Answer"]

        # Ensure we have a full ranking for this test case
        if idx >= len(predictions):
            print(f"Warning: Missing prediction for index {idx}, skipping.")
            continue

        ranked_candidates = predictions[idx]  # Full ranked list

        # Get the rank of the true answer
        if true_answer in ranked_candidates:
            rank = ranked_candidates.index(true_answer) + 1  # 1-based index
        else:
            rank = len(ranked_candidates) + 1  # Worst case if not found

        ranks.append(rank)

        # Compute Hits@K
        for k in k_list:
            if rank <= k:
                hits_at_k[k] += 1

    # Compute final metrics
    num_samples = len(test_data)
    mean_rank = np.mean(ranks)
    mrr = np.mean([1.0 / r for r in ranks])

    hits_at_k = {k: v / num_samples for k, v in hits_at_k.items()}  # Normalize Hits@K

    print("\n===== Link Prediction Evaluation =====")
    print(f"Mean Rank (MR): {mean_rank:.2f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    for k, v in hits_at_k.items():
        print(f"Hits@{k}: {v:.4f}")

    return {"MR": mean_rank, "MRR": mrr, "Hits@K": hits_at_k}


if __name__ == '__main__':
    dataset = "wn18rr"

    # Load test dataset
    with open("/Data/KICGPT/dataset/" + dataset + "/test_answer.txt", "r") as load_f:
        test_triplet = json.load(load_f)

    # Path to saved model outputs
    chat_log_path = f"./outputs/{dataset}/chat_tail.txt"

    # Evaluate using saved predictions
    results = evaluate_link_prediction_from_log(chat_log_path, test_triplet[:2])
