from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
import json
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def init_device():
    # GPU Device
    device = torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"--------\n Using device: {device} \n--------")
    return device

device = init_device()

# Load Model & Tokenizer
def load_model_tokenizer(model_name):
    
    print(f"--------\n Loading model: {model_name} \n--------")
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    print("Model and tokenizer loaded successfully.\n---------------------------")
    
    return model, tokenizer

model_name="apoorvumang/kgt5-base-wikikg90mv2"
model, tokenizer = load_model_tokenizer(model_name=model_name)

# Load Dataset
def load_data(dataset_path):
    print(f"--------\n Loading dataset from {dataset_path} \n--------")

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    print(f"Dataset loaded: {len(dataset)} samples. \n---------------------------")
    
    return dataset

data = load_data(dataset_path="kgt5/data/WN18RR/kgt5_entity_prediction.json")

# Split into Train & Test datasets
def split_data(dataset, test_size):
    
    train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=42)
    print(f"--------\n Training samples: {len(train_set)}, Evaluation samples: {len(test_set)} \n--------")

    return train_set , test_set

train_data , test_data = split_data(data, 0.1)

# Tokenization Function
def tokenize_data(data):
    inputs, outputs = [], []
    for item in tqdm(data, desc="Tokenizing", unit="sample"):
        tokenized_input = tokenizer(item["input"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        tokenized_output = tokenizer(item["output"], padding="max_length", truncation=True, max_length=128, return_tensors="pt").input_ids

        # Append tokenized inputs and labels
        inputs.append({key: val.squeeze(0) for key, val in tokenized_input.items()})
        outputs.append(tokenized_output.squeeze(0))  # Ensure correct tensor shape
        
    print("--------\n Tokenization complete! \n-------- \n---------------------------")
    return inputs, outputs

# Tokenizing datasets
train_inputs, train_outputs = tokenize_data(train_data)
test_inputs, test_outputs = tokenize_data(test_data)

# Convert Tokenized Data into PyTorch Dataset
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

# Create Train & Eval Datasets
train_dataset = KGT5Dataset(train_inputs, train_outputs)
eval_dataset = KGT5Dataset(test_inputs, test_outputs)
print(f"--------\n Dataset prepared: {len(train_dataset)} train samples, {len(eval_dataset)} eval samples.\n--------")

def train_args(train_dataset, eval_dataset):
    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./kgt5_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        logging_dir="./logs",
        fp16=False,  
        dataloader_num_workers=0,  
        gradient_accumulation_steps=2,
        optim="adamw_torch_fused",
    )

    # Create DataLoaders for manual training loop
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)

    # Define Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    return train_dataloader, eval_dataloader, optimizer, loss_fn

train_dataloader, eval_dataloader, optimizer, loss_fn = train_args(train_dataset, eval_dataset)

def train_model():
    # Training Loop 
    print("--------\n Starting training \n--------")
    start_time = time.time()

    train_loss=[]
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        
        for batch in progress_bar:
            optimizer.zero_grad()

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update tqdm progress bar with loss
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes. \n----------------------------")

    # Save the Fine-Tuned Model
    print("---------\n Saving fine-tuned model \n---------")
    model.save_pretrained("kgt5/kgt5_finetuned")
    tokenizer.save_pretrained("kgt5/kgt5_finetuned")

    print("Fine-tuning completed and model saved!\n-----------------------------")

    return num_epochs, train_loss

num_epochs, train_loss = train_model()

# Plot training loss 
def plot_train_loss(num_epochs, train_loss, save_path):
    epochs = list(range(1, num_epochs + 1))   
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker='o', linestyle='-', color='b', label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, format="png", dpi=300)
    print(f"Plot saved as {save_path}")

    plt.show()

plot_train_loss(num_epochs, train_loss, save_path="kgt5/plots/train_loss_plot.png")
