import os
import json
import torch
import tqdm
import random
import transformers
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM

# Fixed random seed for reproducibility
SEED = 42
random.seed(SEED)

prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given a triple from a knowledge graph. Each triple consists of a head entity, a relation, and a tail entity. Please determine the correctness of the triple and response True or False.

### Input:
{}

### Response:

"""

def load_test_dataset(path):
    return json.load(open(path, "r"))

if __name__ == "__main__":
    cuda = "cuda:0"
    # base_model = "NEU-HAI/Llama-2-7b-alpaca-cleaned"
    base_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    lora_weights = f"./lora-{base_model.split('/')[-1]}"


    test_data_path = "data/CoDeX-S-test.json"  # or "data/UMLS-test.json"
    embedding_path = "{}/embeddings.pth".format(lora_weights)

    # 1. Load the test dataset and sample 300 entries (fixed seed).
    full_dataset = load_test_dataset(test_data_path)
    print("Full dataset size: {}".format(len(full_dataset)))

    # Sample 300 random entries for evaluation
    # test_dataset = random.sample(full_dataset, 300)
    test_dataset = full_dataset

    # 2. Load embeddings
    kg_embeddings = torch.load(embedding_path).to(cuda)

    # 3. Prepare model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16
    ).to(cuda)

    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        ignore_mismatched_keys=True,
    ).to(cuda)

    # Setting pad_token_id
    tokenizer.pad_token = tokenizer.eos_token  # Use <eos> as the padding token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config = GenerationConfig(
        pad_token_id=model.config.pad_token_id
    )  # Avoid HF warnings

    # Some LLaMA-based models use 1 and 2 for bos/eos
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model = model.eval()

    # 4. Run inference
    results = []
    for data in tqdm.tqdm(test_dataset, desc="Inference"):
        ent = data["input"]
        ans = data["output"]
        ids = data["embedding_ids"]

        # Convert to torch.LongTensor
        ids = torch.LongTensor(ids).reshape(1, -1).to(cuda)
        # Cast to float16
        prefix = kg_embeddings(ids).half()

        # Prepare text prompt
        prompt = prompt_template.format(ent)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(cuda)

        # Get token embeddings in float16
        token_embeds = model.model.model.embed_tokens(input_ids)
        # Concatenate prefix + token embeddings
        input_embeds = torch.cat((prefix, token_embeds), dim=1)

        # Build attention mask
        prefix_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)
        text_mask = inputs.attention_mask.to(prefix.device)
        attention_mask = torch.cat((prefix_mask, text_mask), dim=1)

        # Generate
        generate_ids = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=16
        )

        # Decode
        context = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        full_response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # Strip the prompt from the response
        response = full_response.replace(context, "").strip()

        # Save full info
        results.append({
            "input": ent,
            "gold": ans,
            "prediction": response,
            "full_llm_output": full_response  # Save the entire generated text
        })

    # 5. Save the 300 inference results to a JSON
    output_path = "inference_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} results to {output_path}")