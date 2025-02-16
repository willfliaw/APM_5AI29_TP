import argparse
import json
import logging
import re
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)

from prompt_selection import Demon_sampler


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
        hf_token = read_hf_token(hf_token_file)
        llm_name = "mistralai/Ministral-8B-Instruct-2410"
        # llm_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        args.reasoning = True

        # Use 4-bit quantization and enable cuDNN benchmark for performance
        quantization_config = BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=True)
        torch.backends.cudnn.benchmark = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            padding_side="left",
            token=hf_token,
            cache_dir="/Data/KICGPT/llm",
        )
        self.tokenizer.use_default_system_prompt = False
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=quantization_config,
            device_map={"": args.device},  # load model on the specified GPU
            torch_dtype=torch.bfloat16,
            token=hf_token,
            cache_dir="/Data/KICGPT/llm",
        )
        self.llm.eval()

        # Modify generation settings based on reasoning argument
        if self.args.reasoning:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_tokens * 10, # double token limit
                do_sample=True,  # Enable sampling
                temperature=0.6,  # Allow some randomness (DeepSeek-R1: 0.5-0.7 recommended)
                top_p=1.0,
                top_k=50,  # Helps prevent extreme randomness
                repetition_penalty=1.1,  # Discourage repetition
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_tokens,
                do_sample=False,  # Fully deterministic
                top_p=1.0,  # No nucleus sampling
                repetition_penalty=1.0,  # No extra penalties
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )


    def create_message(self, input_text, turn_type):
        # Modify the prompt formatting if needed (e.g., remove "I will keep thinking" parts)
        if turn_type == "init_query":
            instruction = self.prompt["init_query"]
            input_text = instruction
        elif turn_type == "first_give_demonstration":
            template = self.prompt["first_give_demonstration"]
            input_text = template.format(question=input_text)
        elif turn_type == "analogy_demonstration":
            template = self.prompt["analogy_demonstration"]
            input_text = template.format(selected_analogy_demonstrations=input_text)
        elif turn_type == "supplement_demonstration":
            template = self.prompt["supplement_demonstration"]
            input_text = template.format(selected_supplement_demonstrations=input_text)
        elif turn_type == "final_query_template":
            template = self.prompt["final_query_template"]
            can_ents, question = input_text
            input_text = template.format(order_of_candidate=can_ents, question=question)
        elif turn_type == "directly_ask":
            template = self.prompt["directly_ask"]
            input_text = template.format(question=input_text)
        else:
            raise NotImplementedError
        message = {"role": "user", "content": input_text}
        return message

    def generate_answer(self, turns):
        # Tokenize input and move to GPU
        input_ids = self.tokenizer.apply_chat_template(turns, return_tensors="pt").to(self.args.device)

        # Explicitly create attention mask to fix the warning
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.args.device)

        # Ensure inference is run without gradients for efficiency
        with torch.no_grad():
            outputs = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Fix for the warning
                generation_config=self.generation_config
            )

        # Decode only the new tokens (generated response)
        answer_tokens = outputs[0, input_ids.shape[1]:-1]


        if args.debug_online:
            print("#########")
            print(f"{turns}")
            print(f"{self.tokenizer.decode(answer_tokens).strip()}")

        return {"role": "assistant", "content": self.tokenizer.decode(answer_tokens).strip()}

    def reset_history(self):
        self.history_messages = []
        self.history_contents = []
        self.token_num = 0

    def load_prompt_template(self, prompt_path, prompt_name):
        with open(prompt_path, "rb") as f:
            prompt = json.load(f)
        return prompt[prompt_name]


class Solver:
    def __init__(self, args):
        self.args = args
        self.LLM = ChatGPT(args, prompt_path=args.prompt_path, prompt_name=args.prompt_name, max_tokens=args.max_tokens)
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
        if args.align_text:
            self.load_align_text()

    def count_token(self, string):
        return len(self.LLM.tokenizer(string)["input_ids"])

    def forward(self, question, tpe):
        self.LLM.reset_history()
        self.reset_history()

        # Prepare candidate answers context
        tpe_str = self.ent2text[tpe]
        candidate_key = "\t".join([str(self.ent2id[tpe]), str(self.rel2id[question])])
        candidate_ids = self.all_candidate_answers[candidate_key]

        candidate_answers = [self.ent2text[self.id2ent[str(id)]] for id in candidate_ids[:self.args.candidate_num]]
        origin_candidates_text = "[" + ", ".join(candidate_answers) + "]"

        # Construct the main question text
        if self.args.query == "tail":
            question_text = self.generate_demonstration_text((tpe_str, question, ""))
        else:
            question_text = self.generate_demonstration_text(("", question, tpe_str))

        # Construct alternating role demonstrations
        demonstration_examples = []
        for step in range(self.args.eff_demon_step):
            analogy_demons, supplement_demons = self.prompt_selector.randomsampler(
                tpe, question, self.args.demon_per_step, step
            )
            combined_demos = analogy_demons + supplement_demons
            demo_text = self.serialize_demonstrations(combined_demos)

            if demo_text.strip() and demo_text.strip() != "None.":
                if step % 2 == 0:
                    demonstration_examples.append({"role": "user", "content": demo_text})
                else:
                    demonstration_examples.append(
                        {"role": "assistant", "content": "Understood, I will use these examples for sorting."})

        # Build final query message
        final_query_message = self.LLM.create_message((origin_candidates_text, question_text), "final_query_template")

        # Assemble full conversation context
        conversation = demonstration_examples + [final_query_message]

        # If reasoning is disabled, add a system prompt
        if not self.args.reasoning:
            system_prompt = {"role": "system",
                             "content": "You are an expert at ranking candidate answers based on examples."}
            conversation.insert(0, system_prompt)  # Ensure system prompt is first

        # If reasoning is enabled, enforce <think> at the start of the response
        if self.args.reasoning:
            conversation[-1]["content"] = conversation[-1]["content"] + " Also please do NOT output lists which do not contain all words from the input even if you think its unnecessary, but output the list containing ALL words WITHOUT quotes seperated by |. Please please, inside of your <think> reasoning, please do NOT analyze each word individually, but try to think about them as groups, as i really want to have a quick answer and your thinking can take quite a while."

        final_response = self.LLM.generate_answer(conversation)

        # Parse and return final ranking
        self.log.append(final_response)
        final_order = self.parse_result(final_response["content"], "final_answer")
        final_order_string = " | ".join(final_order)
        self.log.append(final_order_string)

        return final_order, self.LLM.history_contents, self.log

    def serialize_candidate_answers(self):
        return "[" + ",".join(self.candidate_answers) + "]"

    def check_work_flow(self, response):
        return "no" not in response.lower()

    def relation_text(self, relation, align_text):
        if align_text:
            return self.align_text[relation]
        else:
            parts = relation.strip().replace(".", " ").split("/")
            return " of ".join(reversed([p for p in parts if p])) + " of "

    def serialize_demonstrations(self, demon_triples):
        demon_text = " ".join(self.generate_demonstration_text(tp) + ". " for tp in demon_triples).strip()
        return demon_text if demon_text else "None."

    def generate_demonstration_text(self, triple):
        h, r, t = triple
        # Example: adjust the prompt to remove unnecessary parts
        if self.args.query == "tail":
            demonstration_text = f"Predict the tail entity [MASK] for ({h}, {self.relation_text(r, False)}, [MASK])."
            if t:
                demonstration_text += f" The answer is {t}."
        elif self.args.query == "head":
            demonstration_text = f"Predict the head entity [MASK] for ([MASK], {self.relation_text(r, False)}, {t})."
            if h:
                demonstration_text += f" The answer is {h}."
        return demonstration_text

    def parse_result(self, response, parse_type):
        """
        Parses the response from the LLM, ensuring it always returns a list.
        """
        response = response.lower().strip()

        if parse_type == "final_answer":
            # Step 1: Remove everything before </think>
            if "</think>" in response:
                response = response.split("</think>", 1)[1].strip()

            # Step 2: Identify "final answer" phrases and trim everything before the first one
            final_answer_match = re.search(r"(the final answer is|the final order is|final answer:)", response,
                                           re.IGNORECASE)
            if final_answer_match:
                response = response[final_answer_match.end():].strip()

            # Step 3: Extract content inside the first set of square brackets
            bracket_match = re.search(r"\[([^\]]+)\]", response)
            if bracket_match:
                final_order_raw = bracket_match.group(1).strip()
            else:
                final_order_raw = response  # If no brackets, take the whole response

            # Step 4: Remove any leftover "final order" text before splitting
            final_order_raw = re.sub(r"(the final order is|the final answer is|final order:)", "", final_order_raw,
                                     flags=re.IGNORECASE).strip()

            # Step 5: Split into a list using various reasonable separators
            candidates = re.split(r"\s*\|\s*|\s*,\s*|\s*\n\s*|\s*\t\s*", final_order_raw)

            # Remove duplicates while preserving order
            final_order_list = []
            for candidate in candidates:
                clean_candidate = candidate.strip()
                if clean_candidate and clean_candidate not in final_order_list:
                    final_order_list.append(clean_candidate)

            return final_order_list  # Always return a list

        return []

    def reset_history(self):
        self.log = []
        self.candidate_answers = []
        self.selected_demonstrations = []

    def load_all_candidate_answers(self):
        with open(
            f"/Data/KICGPT/dataset/{self.args.dataset}/retriever_candidate_{self.args.query}.txt", "r"
        ) as load_f:
            self.all_candidate_answers = json.load(load_f)

    def load_align_text(self):
        with open(f"/Data/KICGPT/dataset/{self.args.dataset}/alignment/alignment_clean.txt", "r") as load_f:
            self.align_text = json.load(load_f)

    def load_rel_txt_to_id(self):
        with open(f"/Data/KICGPT/dataset/{self.args.dataset}/get_neighbor/relation2id.txt", "r") as file:
            for line in file.readlines():
                _name, _id = line.strip().split("\t")
                self.rel2id[_name] = _id

    def load_ent_map_id(self):
        with open(f"/Data/KICGPT/dataset/{self.args.dataset}/get_neighbor/entity2id.txt", "r") as file:
            for line in file.readlines():
                _name, _id = line.strip().split("\t")
                self.ent2id[_name] = _id
                self.id2ent[_id] = _name

    def load_ent_to_text(self):
        with open(f"/Data/KICGPT/dataset/{self.args.dataset}/entity2text.txt", "r") as file:
            for line in file.readlines():
                ent, text = line.strip().split("\t")
                self.ent2text[ent] = text


def main(args, all_data):
    output_path = args.output_path
    chat_log_path = args.chat_log_path
    print(f"Start processing {len(all_data)} samples in a single process (batched final queries).")
    solver = Solver(args)
    count = 0
    valid_count = 0

    # Open the output and chat log files once
    with open(output_path, "w") as fout, open(chat_log_path, "w") as flog:
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
            

            sample["Prediction"] = " | ".join(prediction)
            fout.write(json.dumps(sample) + "\n")

            # Convert numeric ground-truth answer to text using ent2text
            ground_truth_text = solver.ent2text.get(sample["Answer"], "UNKNOWN")

            # Convert predictions to properly formatted text
            if isinstance(prediction, list):
                prediction_text = " | ".join(prediction)  # Ensure correct formatting
            else:
                prediction_text = str(prediction)

            # Save correctly formatted predictions
            chat = (
                    str(sample["ID"]) +
                    "\n" +
                    "\n******\n".join(chat_history) +
                    "\nAnswers: " + prediction_text +  # Save ranked predictions as text
                    "\nGround Truth: " + ground_truth_text +  # Save actual ground-truth text
                    "\n------------------------------------------\n"
            )
            flog.write(chat)

    print(f"Processing completed: {valid_count}/{count} samples succeeded.")


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--prompt_name", default="chat")
    parser.add_argument("--bagging_type", default="llm")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0", help="the gpu device")
    parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--demon_per_step", default=8, type=int)
    parser.add_argument("--eff_demon_step", default=10, type=int)
    parser.add_argument("--max_demon_step", default=10, type=int)
    parser.add_argument("--max_llm_input_tokens", default=3750, type=int)
    # Remove multi-process argument to force single-process batch processing.
    args = parser.parse_args()
    args.output_path = f"./outputs/{args.dataset}/output_{args.query}.txt"
    args.chat_log_path = f"./outputs/{args.dataset}/chat_{args.query}.txt"
    print("Start querying the LLM with optimized batching.")
    return args


if __name__ == "__main__":
    args = parse_args()

    args.debug_online = True

    with open(f"/Data/KICGPT/dataset/{args.dataset}/test_answer.txt", "r") as load_f:
        test_triplet = json.load(load_f)
    print(f"Totally {len(test_triplet)} test examples.")
    # If debugging online, limit the number of samples
    if args.debug_online:
        test_triplet = test_triplet[: 5]

    test_triplet = test_triplet[:1000]
    main(args, test_triplet)