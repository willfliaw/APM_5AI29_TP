import json
import random

# Load entity and relation text mappings
def load_mapping(file_path):
    mapping = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping

# Load entity and relation descriptions
entity2text = load_mapping("data/WN18RR/entity2text.txt")
relation2text = load_mapping("data/WN18RR/relation2text.txt")

# Read WN18RR train triples
train_triples = []
with open("data/WN18RR/train.tsv", "r") as f:
    for line in f:
        head, relation, tail = line.strip().split("\t")
        train_triples.append((head, relation, tail))

# Generate KGT5 formatted data
kgt5_data = []
for head, relation, tail in train_triples:
    if head in entity2text and tail in entity2text and relation in relation2text:
        # Head Prediction (predict head entity)
        head_prompt = f"? {relation2text[relation]} {entity2text[tail]}"
        kgt5_data.append({"input": head_prompt, "output": entity2text[head]})

        # Tail Prediction (predict tail entity)
        tail_prompt = f"{entity2text[head]} {relation2text[relation]} ?"
        kgt5_data.append({"input": tail_prompt, "output": entity2text[tail]})

# Shuffle data for randomness
random.shuffle(kgt5_data)

# Save as JSON
with open("kgt5/data/WN18RR/kgt5_entity_prediction.json", "w") as f:
    json.dump(kgt5_data, f, indent=4)

print(f"Saved {len(kgt5_data)} entity prediction prompts to kgt5_entity_prediction.json")
