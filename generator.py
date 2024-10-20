from typing import List, Dict
from cluster import LLMCluster
from prompt.generator_prompt import *
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
import logging
class LLMGenerator:
    def __init__(self, dir = None, num_samples = 5, device = "cpu"):
        if dir == None:
            dir = "/nvme1/wyl/caches/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35/"
        self.num_samples = num_samples
        self.step_count = 0
        # use Qwen2.5-Math-1.5B-Instruct
        self.model = Qwen2ForCausalLM.from_pretrained(dir).to(device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(dir)
        self.cluster = LLMCluster(device=device)
    def get_key_step(self, response: str) -> str:
        splits = response.split("*Step")
        if splits[0][-1] == "*":
            splits[0] = splits[0][:-1]
        return splits[0]

    def generate_responses(self, context: str) -> List[str]:
        responses = []
        input_text = (
            f"{generator_system_prompt}\n"
            # f"{generator_system_response}\n"
            f"{generator_user_prompt}"
            f"{generator_user_response}\n"
            f"{context}"
        )
        logging.info(f"input_text: {input_text}")
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)
        for _ in range(self.num_samples):
            print(f"Generating response: {_}/{self.num_samples}")
            logging.info(f"Generating response: {_}/{self.num_samples}")
            outputs = self.model.generate(inputs, temperature=0.8, max_length = 1024, do_sample = True)[0][inputs.size(1):].to("cpu")
            response = self.tokenizer.decode(outputs, skip_special_tokens=True)
            logging.info(f"response: {response}")
            response = self.get_key_step(response)
            logging.info(f"key step: {response}")
            if response == None:
                response = f"Solution complete.\n"
            responses.append(response)
        return responses


    def cluster_similar_responses(self, context: str, responses: List[str]) -> List[List[str]]:
        clusters = []
        for i, response in enumerate(responses):
            print(f"Clustering response {i}/{len(responses)}")
            logging.info(f"Clustering response {i}/{len(responses)}")
            added_to_cluster = False
            for cluster in clusters:
                if self.cluster.is_same_step(context, cluster[0], response):
                    cluster.append(response)
                    added_to_cluster = True
            if not added_to_cluster:
                clusters.append([response])
        return clusters

    def aggregate_results(self, context:str, responses: List[str]) -> Dict[str, any]:
        final_answers = responses
        clusters = self.cluster_similar_responses(context, final_answers)
        
        cluster_info = []
        for cluster in clusters:
            cluster_info.append({
                "answer": cluster[0],
                "frequency": len(cluster),
                "variants": cluster
            })
        
        cluster_info.sort(key=lambda x: x['frequency'], reverse=True)
        print(cluster_info)
        return {
            "clusters": cluster_info,
            "total_responses": len(responses),
            "num_unique_clusters": len(clusters)
        }

    def evaluate(self, question: str, steps: str, step_count: int) -> Dict[str, any]:
        context = f"{question}" + steps + f"**Step {step_count}**: "
        self.step_count = step_count
        self.cluster.step_count = step_count
        responses = self.generate_responses(context)
        aggregated_result = self.aggregate_results(context, responses)
        
        return aggregated_result

