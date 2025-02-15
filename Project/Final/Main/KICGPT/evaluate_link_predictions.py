import numpy as np


def load_predictions_and_ground_truth(chat_log_path):
    """
    Parses `chat_tail.txt` to extract ranked predictions and ground-truth answers.

    Returns:
    - `predictions`: List of ranked lists (text format)
    - `ground_truths`: List of correct answers (text format)
    """
    predictions, ground_truths = [], []

    with open(chat_log_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("Answers: "):
            ranked_list = line.strip().split("Answers: ")[1].strip().split(" | ")
            predictions.append(ranked_list)
        elif line.startswith("Ground Truth: "):
            ground_truths.append(line.strip().split("Ground Truth: ")[1].strip())

    return predictions, ground_truths


def evaluate_predictions(chat_log_path, k_list=[1, 3, 10]):
    """
    Evaluates link prediction accuracy by comparing ranked predictions with ground truth.

    Metrics:
    - Hits@K
    - Mean Rank (MR)
    - Mean Reciprocal Rank (MRR)
    """
    predictions, ground_truths = load_predictions_and_ground_truth(chat_log_path)

    ranks = []
    hits_at_k = {k: 0 for k in k_list}

    for idx, (ranked_candidates, true_answer) in enumerate(zip(predictions, ground_truths)):
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
    num_samples = len(ground_truths)
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
    chat_log_path = "./outputs/wn18rr/chat_tail.txt"
    results = evaluate_predictions(chat_log_path)
