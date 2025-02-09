import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)


def read_hf_token(file_path):
    """Reads the Hugging Face token from a file."""
    try:
        with open(file_path, "r") as file:
            token = file.read().strip()
            if not token:
                raise ValueError("The token file is empty.")
            return token
    except FileNotFoundError:
        raise FileNotFoundError(f"Token file not found at: {file_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the token: {e}")


class ChatGPT:
    def __init__(self, args, prompt_path, prompt_name, max_tokens):
        self.args = args
        self.history_messages = []
        self.history_contents = []
        self.max_tokens = max_tokens
        self.prompt = self.load_prompt_template(prompt_path, prompt_name)

        hf_token_file = "token.txt"

        # Read the token from the file
        hf_token = read_hf_token(hf_token_file)
        llm_name = "mistralai/Ministral-8B-Instruct-2410"

        # We want to use 4bit quantization to save memory
        quantization_config = BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, padding_side="left", token=hf_token, cache_dir="/Data/KICGPT/llm")
        self.tokenizer.use_default_system_prompt = False
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load LLM.
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=quantization_config,
            device_map={"": 0},  # load all the model layers on GPU 0
            torch_dtype=torch.bfloat16,  # float precision
            token=hf_token,
            cache_dir="/Data/KICGPT/llm",
        )
        self.llm.eval()

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_tokens,
            do_sample=False,
            temperature=0,
            top_p=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def get_response(self, input_text, turn_type):
        if self.args.debug:
            message = self.create_message(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message["content"])
            print("query API to get message:\n%s" % message["content"])
            response = input("input the returned response:")
        else:
            message = self.create_message(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message["content"])
            message = self.query_API_to_get_message(self.history_messages)
            self.history_messages.append(message)
            self.history_contents.append(message["content"])
            response = message["content"].strip()
        return response

    def create_message(self, input_text, turn_type):
        if turn_type == "init_query":
            template = self.prompt["init_query"]
            demonstrations_for_r, r_text = input_text
            input_text = template.format(demonstrations_for_r=demonstrations_for_r, r_text=r_text, r_text2=r_text)
        else:
            raise NotImplementedError
        message = {"role": "user", "content": input_text}
        return message

    def generate_answer(self, turns):
        # Tokenize turns.
        input_ids = self.tokenizer.apply_chat_template(turns, return_tensors="pt").to("cuda")

        # Ensure we don't use gradient to save memory space and computation time.
        with torch.no_grad():
            outputs = self.llm.generate(input_ids, self.generation_config)

        # Recover and decode answer.
        answer_tokens = outputs[0, input_ids.shape[1] : -1]

        return self.tokenizer.decode(answer_tokens).strip()

    def query_API_to_get_message(self, messages):
        while True:
            try:
                res = self.generate_answer(messages)
                if args.debug_online:
                    print(res)
                return res
            except Exception as e:
                print(f"An exception occurred: {e}")
                time.sleep(20)

    def reset_history(self):
        self.history_messages = []
        self.history_contents = []

    def reset_history_messages(self):
        self.history_messages = []

    def reset_history_contents(self):
        self.history_contents = []

    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]


class Solver:
    def __init__(self, args):
        self.args = args
        self.LLM = ChatGPT(
            args=args,
            prompt_path=args.prompt_path,
            prompt_name=args.prompt_name,
            max_tokens=args.max_tokens,
        )

        self.log = []

        self.ent2text = defaultdict(str)
        self.rel2text = defaultdict(str)
        self.load_ent_to_text()
        self.load_rel_to_text()

    def forward(self, r, triples):  # Here tpe_id not a int id, but rather '/m/08966'
        self.LLM.reset_history()
        self.reset_history()
        demonstration_text = self.serialize_demonstrations(triples)
        r_text = self.relation_text(r)
        final_response = self.LLM.get_response((demonstration_text, r_text), "init_query")
        self.log.append(final_response)

        return final_response, self.LLM.history_contents

    def relation_text(self, relation):
        if args.dataset == "wn18rr":
            return self.rel2text[relation]
        else:
            relation_hierachy_list = relation.strip().replace(".", " ").split("_")
            final_string = ""
            for st in reversed(relation_hierachy_list):
                if st != "":
                    final_string += st + " of "
            return final_string

    def serialize_demonstrations(self, demon_triples):
        demon_text = ""
        for tp in demon_triples:
            demon_text += self.generate_demonstration_text(tp) + "\n"
        demon_text.strip()
        if demon_text == "":
            demon_text = "None."
        return demon_text

    def generate_demonstration_text(self, triple):
        h, r, t = triple
        h = self.ent2text[h]
        t = self.ent2text[t]
        if args.dataset == "wn18rr":
            demonstration_text = t + " " + self.relation_text(r) + " " + h
        else:
            demonstration_text = t + " is the " + self.relation_text(r) + h
        return demonstration_text

    def reset_history(self):
        self.log = []

    def load_ent_to_text(self):
        with open("/Data/KICGPT/dataset/" + args.dataset + "/entity2text.txt", "r", encoding="utf8") as file:
            entity_lines = file.readlines()
            for line in entity_lines:
                ent, text = line.strip().split("\t")
                self.ent2text[ent] = text

    def load_rel_to_text(self):
        with open("/Data/KICGPT/dataset/" + args.dataset + "/relation2text.txt", "r") as file:
            rel_lines = file.readlines()
            for line in rel_lines:
                rel, text = line.strip().split("\t")
                self.rel2text[rel] = text


def main(args, demonstration_r, idx):
    if idx == -1:
        output_path = args.output_path
        chat_log_path = args.chat_log_path
    else:
        idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
        output_path = args.output_path + "_" + idx
        chat_log_path = args.chat_log_path + "_" + idx

    print("Start PID %d and save to %s" % (os.getpid(), chat_log_path))
    solver = Solver(args)

    with open(output_path, "w") as f:
        with open(chat_log_path, "w") as fclog:
            for key in demonstration_r:
                try:
                    r = key
                    triples = random.sample(demonstration_r[r], args.demon_per_r)
                    clean_relation, chat_history = solver.forward(r, triples)
                except Exception as e:
                    print(e)
                    logging.exception(e)
                    continue

                clean_text = defaultdict(str)
                clean_text["Raw"] = key
                clean_text["Description"] = clean_relation
                f.write(json.dumps(clean_text) + "\n")

                chat = str(key) + "\n" + "\n******\n".join(chat_history) + "\n------------------------------------------\n"
                fclog.write(chat)

    print("---------------PID %d end--------------" % (os.getpid()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wn18rr")
    parser.add_argument("--output_path", default="/Data/KICGPT/dataset/wn18rr/alignment/alignment_output.txt")
    parser.add_argument("--chat_log_path", default="/Data/KICGPT/dataset/wn18rr/alignment/alignment_chat.txt")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_online", action="store_true")

    parser.add_argument("--max_tokens", default=300, type=int, help="max-token")
    parser.add_argument("--prompt_path", default="./prompts/text_alignment.json")
    parser.add_argument("--prompt_name", default="chat")
    parser.add_argument("--bagging_type", default="llm")

    parser.add_argument("--device", default=0, help="the gpu device")

    parser.add_argument("--demon_per_r", default=30)
    parser.add_argument("--num_process", default=1, type=int, help="the number of multi-process")

    args = parser.parse_args()

    print("Start querying the LLM.")
    return args


if __name__ == "__main__":
    args = parse_args()

    demonstration_r = defaultdict(list)
    with open("/Data/KICGPT/dataset/" + args.dataset + "/demonstration/all_r_triples.txt", "r") as f:
        demonstration_r = json.load(f)

    if args.num_process == 1:
        main(args, demonstration_r, idx=-1)
    else:
        pass
