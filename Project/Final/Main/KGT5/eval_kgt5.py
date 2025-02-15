from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import gc
import json
import numpy as np
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Initialize device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Using device: {device}")

# Load model & tokenizer
def load_model_tokenizer(model_path):
    print(f"--------\n Loading model: {model_path} \n--------")
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    return model, tokenizer

model_path = "kgt5/kgt5_finetuned"
model, tokenizer = load_model_tokenizer(model_path)
model.eval()

# Load dataset
def load_data(dataset_path):
    with open(dataset_path, "r") as f:
        return json.load(f)

dataset_path = "kgt5/data/WN18RR/kgt5_entity_prediction.json"
data = load_data(dataset_path)
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# Tokenize test data
def tokenize_data(data):
    inputs, outputs = [], []
    for item in tqdm(data, desc="Tokenizing", unit="sample"):
        tokenized_input = tokenizer(item["input"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        tokenized_output = tokenizer(item["output"], padding="max_length", truncation=True, max_length=128, return_tensors="pt").input_ids
        inputs.append({key: val.squeeze(0) for key, val in tokenized_input.items()})
        outputs.append(tokenized_output.squeeze(0))
    return inputs, outputs

test_inputs, test_outputs = tokenize_data(test_data)

# Dataset class
class KGT5Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx]["input_ids"],
            "attention_mask": self.inputs[idx]["attention_mask"],
            "labels": self.outputs[idx],
        }

# Create test dataset & dataloader
test_dataset = KGT5Dataset(test_inputs, test_outputs)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate model
def evaluate_model(model, dataloader):
    hits_at_k = {f"Hits@{k}": 0 for k in [1, 3, 10]}
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Ensure labels are in proper format
            labels = labels.detach().cpu().numpy()

            # Create decoder input IDs
            decoder_input_ids = decoder_input_ids = model._shift_right(torch.tensor(labels, dtype=torch.long, device=device))

            # Forward pass with decoder_input_ids
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            decoder_input_ids=decoder_input_ids)
            
            logits = outputs.logits.detach().cpu().numpy()

            # Compute top-k predictions
            predictions = np.argsort(-logits, axis=-1)  # Sort in descending order

            batch_size, seq_length, _ = logits.shape  # Extract batch size & sequence length
            total_samples += batch_size

            for i in range(batch_size):  
                for j in range(seq_length):  # Iterate over each token position
                    true_label = labels[i, j]  # Get the true token label
                    if true_label == tokenizer.pad_token_id:  # Ignore padding tokens
                        continue
                    
                    for k in [1, 3, 10]:  # Check if the true label appears in top-k predictions
                        if true_label in predictions[i, j, :k]:  
                            hits_at_k[f"Hits@{k}"] += 1

            torch.cuda.empty_cache()
            gc.collect()

    # Compute final evaluation results
    eval_results = {key: val / total_samples for key, val in hits_at_k.items()}
    return eval_results

# Run evaluation
eval_results = evaluate_model(model, test_dataloader)
print("\nEvaluation Results:")
print(json.dumps(eval_results, indent=4))

# Save results
with open("kgt5/results/eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=4)

# Plot Hits@K
def plot_hits_at_k(metrics, save_path):
    k_values = [int(k.split("@")[1]) for k in metrics.keys()]
    hits_values = [metrics[f"Hits@{k}"] for k in k_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, hits_values, marker='o', linestyle='-', color='b', label="Hits@K")
    plt.xlabel("K values")
    plt.ylabel("Score")
    plt.title("Hits@K Evaluation")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, format="png", dpi=300)
    print(f"Plot saved as {save_path}")
    plt.show()

# Generate and save Hits@K plot
plot_hits_at_k(eval_results, save_path="kgt5/plots/hits_at_k.png")
