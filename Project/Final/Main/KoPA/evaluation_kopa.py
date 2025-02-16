import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def load_inference_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    # 1. Load results
    inference_results_path = "codex_inference_results_alpaca.json"
    data = load_inference_results(inference_results_path)

    # 2. Collect predictions and gold labels
    y_true = []
    y_pred = []

    for item in data:
        gold = item["gold"]
        pred = item["prediction"]

        # Convert "True"/"False" to 1/0 if that is the consistent format
        true_label = 1 if "True" in gold else 0
        pred_label = 1 if "True" in pred else 0

        y_true.append(true_label)
        y_pred.append(pred_label)

    # 3. Compute metrics
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 4. Print out
    print("Evaluation Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1-score:  {f1:.4f}")