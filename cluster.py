from prompt.cluster_prompt import *
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
from utils.utils import *
from difflib import SequenceMatcher
from typing import List

class LLMCluster:
    def __init__(self, dir = None, device = "cpu"):
        if dir == None:
            dir = "/nvme1/wyl/caches/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
        self.model = Qwen2ForCausalLM.from_pretrained(dir).to(device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(dir)
        self.step_count = 0
        self.device = device
    def is_same_step(self, content: str, a: str, b: str) -> bool:
        prompt = content + f"**[1]**: " + a
        prompt = prompt + f"**[2]**: " + b
        prompt += "\n Are the two conclusions the same? "
        model_inputs = get_inputs(self.tokenizer, cluster_sys_prompt3, prompt).to(self.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=3)
        response = get_response(self.tokenizer, model_inputs, generated_ids)[0]
        return "Same" in response or "same" in response

    def cluster(self, context: str, responses: List[str]) -> List[List[str]]:
        prompt = context + "\n Possibilities:\n"
        for i, response in enumerate(responses):
            prompt += f"**[{i + 1}]**: {response}\n"
        prompt += "Cluster these responses: "
        model_inputs = get_inputs(self.tokenizer, cluster_sys_prompt2, prompt).to(self.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=50)
        ans = get_response(self.tokenizer, model_inputs, generated_ids)[0]
        clusters = get_clusters(ans, responses)
        return clusters

    def extract(self, response: str) -> str:
        model_inputs = get_inputs(self.tokenizer, extract_sys_prompt, response, "The key conclusion of the step is: ").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512).to(self.device)
        response = get_response(self.tokenizer, model_inputs, generated_ids)[0]
        return response
    
    def pair_wise_cluster(self, question: str, conclusions: List[str]) -> List[List[str]]:
        clusters = []
        for _, response in enumerate(conclusions):
            print(_, "/", len(conclusions))
            added_to_cluster = False
            for cluster in clusters:
                if self.is_same_step(question, response, cluster[0]):
                    cluster.append(response)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([response])
        return clusters


class NaiveCluster:
    def __init__(self, threshold = 0.8):
        self.threshold = threshold
    def cluster(self, context: str, responses: List[str]) -> List[List[str]]:
        clusters = []
        for _, response in enumerate(responses):
            added_to_cluster = False
            for cluster in clusters:
                if SequenceMatcher(None, response, cluster[0]).ratio() > self.threshold:
                    cluster.append(response)
                    added_to_cluster = True
            if not added_to_cluster:
                clusters.append([response])
        return clusters