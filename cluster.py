from prompt.cluster_prompt import *
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
import time
from difflib import SequenceMatcher
from typing import List

class LLMCluster:
    def __init__(self, dir = None, device = "cpu"):
        # use Qwen2.5-7B-Instruct
        if dir == None:
            dir = "/nvme1/wyl/caches/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
        self.model = Qwen2ForCausalLM.from_pretrained(dir).to(device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(dir)
        self.step_count = 0

    def is_same_step(self, content: str, a: str, b: str) -> bool:
        prompt = content + f"Possibility 1: " + a
        prompt = prompt + f"Possibility 2: " + b
        prompt += "\nDo possibility 1, 2 get the same conclusion? Response: "
        messages = [
            {"role": "system", "content": cluster_system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=3)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return "Same" in response or "same" in response

    def cluster(self, context: str, responses: List[str]) -> List[List[str]]:
        prompt = context + "\n Possibilities:\n"
        for i, response in enumerate(responses):
            prompt += f"**[{i + 1}]**: {response}\n"
        prompt += "Cluster these responses: "
        messages = [
            {"role": "system", "content": cluster_system_prompt2},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=50)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("!", len(responses), response)
        clusters = []
        for cluster in response.split("**[")[1:]:
            cluster = cluster.split("]**")[0]
            if len(cluster) == 0:
                continue
            cluster = list(map(int, cluster.split(",")))
            lst = []
            for i in range(len(cluster)):
                if cluster[i] <= len(responses):
                    lst.append(responses[cluster[i] - 1])
                else:
                    raise ValueError(f"Invalid cluster index {cluster[i]}")
            if len(lst) > 0:
                clusters.append(lst)
        return clusters

class NaiveCluster:
    def __init__(self, threshold = 0.8):
        self.threshold = threshold
    def is_same_step(self, content: str, a: str, b: str) -> bool:
        return SequenceMatcher(None, a, b).ratio() > self.threshold