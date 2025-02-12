import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import time
import traceback
from collections import defaultdict

import torch
from prompt_selection import Demon_sampler
from tqdm import tqdm
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
        logging.exception(e)
        raise RuntimeError(f"An error occurred while reading the token: {e}")


class ChatGPT:
    def __init__(self, args, prompt_path, prompt_name, max_tokens):
        self.args = args
        self.history_messages = []
        self.history_contents = []
        self.max_tokens = max_tokens
        self.prompt = self.load_prompt_template(prompt_path, prompt_name)
        self.token_num = 0

        hf_token_file = "token.txt"

        # Read the token from the file
        hf_token = read_hf_token(hf_token_file)
        llm_name = "mistralai/Ministral-8B-Instruct-2410"

        # We want to use 4bit quantization to save memory
        quantization_config = BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=True)

        # Consider adding these optimizations:
        torch.backends.cudnn.benchmark = True  # If input sizes are consistent

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
            # temperature=0,
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

    def query_localLLM_to_get_response(self, message):
        # input: message: {role': 'user', 'content': string(input_text to LLM, which has implemented) }
        # return:  response: {role': 'assistant', 'content': string(output_text wich need you to fetch and store here)}
        output_text = ""  # modifiy here
        response = {"role": "assistant", "content": output_text}
        if output_text == "":
            print("Implement The function")
        return response

    def create_message(self, input_text, turn_type):
        if turn_type == "init_query":
            instruction = self.prompt["init_query"]
            input_text = instruction

        elif turn_type == "first_give_demonstration":
            template = self.prompt["first_give_demonstration"]
            question = input_text
            input_text = template.format(question=question)
        elif turn_type == "analogy_demonstration":
            template = self.prompt["analogy_demonstration"]
            analogy_demons = input_text
            input_text = template.format(selected_analogy_demonstrations=analogy_demons)
        elif turn_type == "supplement_demonstration":
            template = self.prompt["supplement_demonstration"]
            supplement_demons = input_text
            input_text = template.format(selected_supplement_demonstrations=supplement_demons)
        elif turn_type == "final_query_template":
            template = self.prompt["final_query_template"]
            can_ents, question = input_text
            input_text = template.format(order_of_candidate=can_ents, question=question)
        elif turn_type == "directly_ask":
            template = self.prompt["directly_ask"]
            question = input_text
            input_text = template.format(question=question)
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

        return {"role": "assistant", "content": self.tokenizer.decode(answer_tokens).strip()}

    def query_API_to_get_message(self, messages):
        while True:
            try:
                res = self.generate_answer(messages)
                if args.debug_online:
                    print(messages)
                    print(res)
                return res
            except Exception as e:
                print(f"An error occurred: {e}")
                logging.exception(e)
                time.sleep(20)

    def reset_history(self):
        self.history_messages = []
        self.history_contents = []
        self.token_num = 0

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
        self.max_llm_input_token = args.max_llm_input_tokens
        self.prompt_selector = Demon_sampler(args)

        self.log = []
        self.candidate_answers = []
        self.selected_demonstrations = []

        self.id2ent = defaultdict(str)
        self.ent2id = defaultdict(str)
        self.rel2id = defaultdict(str)
        self.ent2text = defaultdict(str)
        self.all_candidate_answers = defaultdict(list)
        self.align_text = defaultdict(str)

        self.load_rel_txt_to_id()
        self.load_ent_map_id()
        self.load_all_candidate_answers()
        self.load_ent_to_text()
        if self.args.align_text:
            self.load_align_text()

    def count_token(self, string):
        return len(self.LLM.tokenizer(string)["input_ids"])

    def forward(self, question, tpe):  # Here tpe_id not a int id, but like '/m/08966'
        self.LLM.reset_history()
        self.reset_history()
        tpe_str = self.ent2text[tpe]
        candidate_ids = self.all_candidate_answers["\t".join([str(self.ent2id[tpe]), str(self.rel2id[question])])]
        for id in candidate_ids[: args.candidate_num]:
            self.candidate_answers.append(self.ent2text[self.id2ent[str(id)]])
        origin_candidates_text = self.serialize_candidate_answers()
        if args.query == "tail":
            question_text = self.generate_demonstration_text((tpe_str, question, ""))
        elif args.query == "head":
            question_text = self.generate_demonstration_text(("", question, tpe_str))
        query_token_num = self.count_token(self.LLM.create_message((origin_candidates_text, question_text), "final_query_template")["content"])

        init_response = self.LLM.get_response((""), "init_query")
        assert self.check_work_flow(init_response), "LLM Not Understand Task"

        effective_demon_step = 0
        current_demon_step = -1

        while effective_demon_step < args.eff_demon_step and current_demon_step < args.max_demon_step:
            if current_demon_step == -1:
                current_demon_response = self.LLM.get_response((question_text), "first_give_demonstration")
                current_demon_step += 1
                true_demons = self.prompt_selector.true_candidate_v2(tpe, question, num=args.demon_per_step // 2)
                true_demon_text = self.serialize_demonstrations(true_demons)
                if true_demon_text != "None.":
                    current_demon_response = self.LLM.get_response((true_demon_text), "analogy_demonstration")
                if self.LLM.token_num >= args.max_llm_input_tokens - query_token_num:
                    self.LLM.history_messages.pop()
                    self.LLM.history_messages.pop()
                    self.LLM.history_contents.pop()
                    self.LLM.history_contents.pop()
                    break
                continue
            analogy_demons, supplement_demons = self.prompt_selector.randomsampler(tpe, question, args.demon_per_step, current_demon_step)
            analogy_demon_text = self.serialize_demonstrations(analogy_demons)
            supplement_demon_text = self.serialize_demonstrations(supplement_demons)
            if analogy_demon_text == "None." and supplement_demon_text == "None.":
                break

            if analogy_demon_text != "None":
                current_demon_response = self.LLM.get_response((analogy_demon_text), "analogy_demonstration")
                if self.LLM.token_num >= args.max_llm_input_tokens - query_token_num:
                    self.LLM.history_messages.pop()
                    self.LLM.history_messages.pop()
                    self.LLM.history_contents.pop()
                    self.LLM.history_contents.pop()
                    break
            if supplement_demon_text != "None.":
                current_demon_response = self.LLM.get_response((supplement_demon_text), "supplement_demonstration")
                if self.LLM.token_num >= args.max_llm_input_tokens - query_token_num:
                    self.LLM.history_messages.pop()
                    self.LLM.history_messages.pop()
                    self.LLM.history_contents.pop()
                    self.LLM.history_contents.pop()
                    break
            current_demon_step += 1

            if self.check_work_flow(current_demon_response):
                effective_demon_step += 1

            self.log.append(f"demonstration: {effective_demon_step:02d}/{current_demon_step:02d} step")
            print(f"demonstration: {effective_demon_step:02d}/{current_demon_step:02d} step")

        final_response = self.LLM.get_response((origin_candidates_text, question_text), "final_query_template")

        self.log.append(final_response)
        final_order = self.parse_result(final_response, "final_answer")
        self.log.append(final_order)

        return final_order, self.LLM.history_contents, self.log

    def serialize_candidate_answers(self):
        candidiate_str = "[" + ",".join(self.candidate_answers) + "]"
        return candidiate_str

    def check_work_flow(self, response):
        if "no" in response.lower():
            return False
        return True

    def relation_text(self, relation, align_text):
        if align_text:
            return self.align_text[relation]
        else:
            relation_hierachy_list = relation.strip().replace(".", " ").split("/")
            final_string = ""
            for st in reversed(relation_hierachy_list):
                if st != "":
                    final_string += st + " of "
            return final_string

    def serialize_demonstrations(self, demon_triples):
        demon_text = ""
        for tp in demon_triples:
            demon_text += self.generate_demonstration_text(tp) + ". "
        demon_text.strip()
        if demon_text == "":
            demon_text = "None."
        return demon_text

    def generate_demonstration_text(self, triple):
        h, r, t = triple
        demonstration_text = ""
        if self.args.query == "tail":
            if self.args.align_text:
                demonstration_text = "predict the tail entity [MASK] from the given ("
                demonstration_text += h + ", " + self.relation_text(r, False)
                demonstration_text += ', [MASK]) by completing the sentence "'
                demonstration_text += self.relation_text(r, True).replace("[H]", h).replace("[T]", "[the answer]") + '? The answer is "'
                if t != "":
                    demonstration_text += ". The answer is " + t + ", so the [MASK] is " + t
            else:
                demonstration_text = "predict the tail entity [MASK] from the given ("
                demonstration_text += h + ", " + self.relation_text(r, False)
                demonstration_text += ', [MASK]) by completing the sentence "what is the '
                demonstration_text += self.relation_text(r, False) + h + '? The answer is "'
                if t != "":
                    demonstration_text += ". The answer is " + t + ", so the [MASK] is " + t
        elif self.args.query == "head":
            if self.args.align_text:
                demonstration_text = "predict the head entity [MASK] from the given ("
                demonstration_text += "[MASK]" + ", " + self.relation_text(r, False)
                demonstration_text += ", " + t + ') by completing the sentence "'
                demonstration_text += self.relation_text(r, True).replace("[H]", "[the answer]").replace("[T]", t) + '? The answer is "'
                if h != "":
                    demonstration_text += ". The answer is " + h + ", so the [MASK] is " + h
            else:
                demonstration_text = "predict the head entity [MASK] from the given ("
                demonstration_text += "[MASK]" + ", " + self.relation_text(r, False)
                demonstration_text += ", " + t + ') by completing the sentence "' + t + " is the "
                demonstration_text += self.relation_text(r, False) + "what" + '? The answer is "'
                if h != "":
                    demonstration_text += ". The answer is " + h + ", so the [MASK] is " + h
        return demonstration_text

    def parse_result(self, response, parse_type):
        response = response.lower()
        if parse_type == "final_answer":
            if "the final order:" in response:
                final_order_raw = re.split("the final order:", response)[1].strip().strip(".").strip("\[").strip("\]")
                final_order_raw_list = final_order_raw.split(" | ")
                final_order_list = []
                for candidate in final_order_raw_list:
                    if candidate not in final_order_list:
                        final_order_list.append(candidate)
                final_order = " | ".join(final_order_list)
        return final_order

    def reset_history(self):
        self.log = []
        self.candidate_answers = []
        self.selected_demonstrations = []

    def load_all_candidate_answers(self):
        with open(
            "/Data/KICGPT/dataset/" + self.args.dataset + "/retriever_candidate_" + args.query + ".txt",
            "r",
        ) as load_f:
            self.all_candidate_answers = json.load(load_f)

    def load_align_text(self):
        with open("/Data/KICGPT/dataset/" + self.args.dataset + "/alignment/alignment_clean.txt", "r") as load_f:
            self.align_text = json.load(load_f)

    def load_rel_txt_to_id(self):
        with open("/Data/KICGPT/dataset/" + self.args.dataset + "/get_neighbor/relation2id.txt", "r") as file:
            relation_lines = file.readlines()
            for line in relation_lines:
                _name, _id = line.strip().split("\t")
                self.rel2id[_name] = _id

    def load_ent_map_id(self):
        with open("/Data/KICGPT/dataset/" + self.args.dataset + "/get_neighbor/entity2id.txt", "r") as file:
            entity_lines = file.readlines()
            for line in entity_lines:
                _name, _id = line.strip().split("\t")
                self.ent2id[_name] = _id
                self.id2ent[_id] = _name

    def load_ent_to_text(self):
        with open("/Data/KICGPT/dataset/" + self.args.dataset + "/entity2text.txt", "r") as file:
            entity_lines = file.readlines()
            for line in entity_lines:
                ent, text = line.strip().split("\t")
                self.ent2text[ent] = text


def main(args, all_data, idx):
    if idx == -1:
        output_path = args.output_path
        chat_log_path = args.chat_log_path
    else:
        idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
        output_path = args.output_path + "_" + idx
        chat_log_path = args.chat_log_path + "_" + idx

    print("Start PID %d and save to %s" % (os.getpid(), output_path))
    solver = Solver(args)

    count = 0
    valid_count = 0
    with open(output_path, "w") as f:
        with open(chat_log_path, "w") as fclog:
            for sample in tqdm(all_data, total=len(all_data)):
                count += 1
                try:
                    tpe = sample["HeadEntity"] if args.query == "tail" else sample["Answer"]
                    question = sample["Question"]

                    prediction, chat_history, record = solver.forward(question, tpe)
                    valid_count += 1
                except Exception as e:
                    print(f"An error occurred: {e}")
                    logging.exception(e)
                    continue

                chat = (
                    str(sample["ID"])
                    + "\n"
                    + "\n******\n".join(chat_history)
                    + "\nAnswers: "
                    + str(sample["Answer"])
                    + "\n------------------------------------------\n"
                )
                fclog.write(chat)

                sample["Prediction"] = prediction
                f.write(json.dumps(sample) + "\n")

    print("---------------PID %d end with %d/%d samples--------------" % (os.getpid(), valid_count, count))


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", default="fb15k-237")
    parser.add_argument("--dataset", default="wn18rr")

    parser.add_argument("--candidate_num", default=50, type=int)
    parser.add_argument("--output_path", default="./outputs/wn18rr/output_tail.txt")
    parser.add_argument("--chat_log_path", default="./outputs/wn18rr/chat_tail.txt")
    parser.add_argument("--query", default="tail", required=False)
    parser.add_argument("--model_path", default=None)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_online", action="store_true")
    parser.add_argument("--align_text", action="store_true")

    parser.add_argument("--max_tokens", default=300, type=int, help="max-token")
    parser.add_argument("--prompt_path", default="./prompts/link_prediction.json")
    parser.add_argument(
        "--prompt_name",
        default="chat",
    )
    parser.add_argument(
        "--bagging_type",
        default="llm",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=0, help="the gpu device")

    parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--demon_per_step", default=8)
    parser.add_argument("--eff_demon_step", default=10)
    parser.add_argument("--max_demon_step", default=10)
    parser.add_argument("--max_llm_input_tokens", default=3750, type=int)
    parser.add_argument("--num_process", default=1, type=int, help="the number of multi-process")

    args = parser.parse_args()
    args.output_path = "./outputs/" + args.dataset + "/output_" + args.query + ".txt"
    args.chat_log_path = "./outputs/" + args.dataset + "/chat_" + args.query + ".txt"
    print("Start querying the LLM.")
    return args


if __name__ == "__main__":
    args = parse_args()

    # Manual args setting
    args.debug_online = True
    args.query = "tail"

    test_triplet = []

    with open("/Data/KICGPT/dataset/" + args.dataset + "/test_answer.txt", "r") as load_f:
        test_triplet = json.load(load_f)
    print("Totally %d test examples." % len(test_triplet))

    if args.debug_online:
        test_triplet = test_triplet[0 : 2 * args.num_process]
    if args.num_process == 1:
        main(args, test_triplet, idx=-1)
    else:
        num_each_split = int(len(test_triplet) / args.num_process)
        p = mp.Pool(args.num_process)
        for idx in range(args.num_process):
            start = idx * num_each_split
            if idx == args.num_process - 1:
                end = max((idx + 1) * num_each_split, len(test_triplet))
            else:
                end = (idx + 1) * num_each_split
            split_data = test_triplet[start:end]
            try:
                p.apply_async(main, args=(args, split_data, idx))
            except Exception as e:
                logging.exception(e)

        p.close()
        p.join()
        print("All of the child processes over!")

# Debug:
#       python3 link_prediction.py --dataset fb15k-237   --debug --query tail
#       python3 link_prediction.py --dataset fb15k-237   --debug --query head
# debug_online:
#       python3 link_prediction.py --dataset fb15k-237 --debug_online --query tail
#       python3 link_prediction.py --dataset fb15k-237 --debug_online --query head
