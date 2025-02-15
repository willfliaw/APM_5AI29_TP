import os
import json
import torch
import transformers
from peft import PeftModel, PeftConfig
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM


prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given a triple from a knowledge graph. Each triple consists of a head entity, a relation, and a tail entity. Please determine the correctness of the triple and response True or False.

### Input:
{}

### Response:

"""

def load_test_dataset(path):
    test_dataset = json.load(open(path, "r"))
    return test_dataset

if __name__ == "__main__":
    cuda = "cuda:0"
    # base_model = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    base_model = "NEU-HAI/Llama-2-7b-alpaca-cleaned"
    lora_weights = f"./lora-{base_model.split('/')[-1]}"
    # test_data_path = "data/UMLS-test.json"
    test_data_path = "data/CoDeX-S-test.json"
    embedding_path = "{}/embeddings.pth".format(lora_weights)
    test_dataset = load_test_dataset(test_data_path)[2000:2020]
    kg_embeddings = torch.load(embedding_path).to(cuda)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16
    ).to(cuda)
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        ignore_mismatched_keys=True, # Keys like these are missing, but i cannot see an issue ['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight', 'base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight', 'base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight'
    ).to(cuda)
    # unwind broken decapoda-research config
    tokenizer.pad_token = tokenizer.eos_token  # Use <eos> as the padding token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config = GenerationConfig(pad_token_id=model.config.pad_token_id)  # Avoids HF warnings
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model = model.eval()
    result = []
    for data in test_dataset:
        ent = data["input"]
        ans = data["output"]
        ids = data["embedding_ids"]

        # 1) The prefix from your embeddings is Float32 by default; cast it to Half:
        ids = torch.LongTensor(ids).reshape(1, -1).to(cuda)
        prefix = kg_embeddings(ids).half()  # <-- Cast to float16

        # 2) Prepare the text prompt as usual:
        prompt = prompt_template.format(ent)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(cuda)

        # 3) Get token embeddings (already in half if your model is float16):
        token_embeds = model.model.model.embed_tokens(input_ids)  # This is float16

        # 4) Concat prefix + token embeddings => same dtype now (both half):
        input_embeds = torch.cat((prefix, token_embeds), dim=1)

        # 5) Build an attention mask for the new input_embeds:
        #    - "prefix" has shape [batch_size, prefix_length, hidden_size]
        #    - "token_embeds" has shape [batch_size, seq_len, hidden_size]
        #    => We need an attention mask of shape [batch_size, prefix_length + seq_len]
        prefix_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)  # shape [1, prefix_length]
        text_mask = inputs.attention_mask.to(prefix.device)  # shape [1, seq_len]
        attention_mask = torch.cat((prefix_mask, text_mask), dim=1)  # shape [1, prefix_length + seq_len]

        # 6) Pass the attention_mask to model.generate():
        generate_ids = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,  # <-- Provide attention mask
            max_new_tokens=16
        )

        # 7) Decode, strip the prompt from response, etc. (unchanged)
        context = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.replace(context, "").strip()

        print(f"################ {ans} {ent}")
        print(response + '\n')
        result.append(
            {
                "answer": ans,
                "predict": response
            }
        )
    answer = []
    predict = []
    for data in result:
        if "True" in data["answer"]:
            answer.append(1)
        else:
            answer.append(0)
        if "True" in data["predict"]:
            predict.append(1)
        else:
            predict.append(0)
    acc = accuracy_score(y_true=answer, y_pred=predict)
    p = precision_score(y_true=answer, y_pred=predict)
    r = recall_score(y_true=answer, y_pred=predict)
    f1 = f1_score(y_true=answer, y_pred=predict)
    print(acc, p, r, f1)
    

    
