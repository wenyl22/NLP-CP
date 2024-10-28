from typing import List, Dict
from cluster import LLMCluster, NaiveCluster
from prompt.generator_prompt import *
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
class LLMGenerator:
    def __init__(self, dir = None, num_samples = 5, device = "cpu"):
        if dir == None:
            dir = "/nvme1/wyl/caches/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35/"
        self.num_samples = num_samples
        self.step_count = 0
        self.model = Qwen2ForCausalLM.from_pretrained(dir).to(device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(dir)
        self.cluster = {
            "llm": LLMCluster(device=device),
            "naive": NaiveCluster(threshold=0.8)
        }
    def get_key_step(self, response: str) -> str:
        splits = response.split(f"{self.step_count + 1}. **")
        return splits[0]

    def generate_responses(self, text: str) -> List[str]:
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512, do_sample = True, num_return_sequences = self.num_samples)
        generated_ids = [output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for _, response in enumerate(responses):
            response = self.get_key_step(response)
            response = f"{self.step_count}. **{response}"
            responses[_] = response
        return responses

    def cluster_similar_responses(self, context: str, responses: List[str]) -> List[List[str]]:
        clusters = []
        if self.mode == "naive":
            for i, response in enumerate(responses):
                added_to_cluster = False
                for cluster in clusters:
                    if self.cluster["naive"].is_same_step(context, cluster[0], response):
                        cluster.append(response)
                        added_to_cluster = True
                if not added_to_cluster:
                    clusters.append([response])
        elif self.mode == "llm":
            clusters = self.cluster["llm"].cluster(context, responses)
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
        #print(cluster_info)
        return {
            "clusters": cluster_info,
            "total_responses": len(responses),
            "num_unique_clusters": len(clusters)
        }

    def evaluate(self, question: str, steps: str, step_count: int, mode: str) -> Dict[str, any]:
        self.mode = mode
        self.step_count = step_count
        self.cluster[mode].step_count = step_count
        messages = [
            {"role": "system", "content": generator_system_prompt},
            {"role": "user", "content": question}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
        context = text + "\nThe step-by-step solution is as follows:\n" + steps + f"{step_count}. **"
        responses = self.generate_responses(context)
        context = text + "\nThe step-by-step solution is as follows:\n" + steps
        aggregated_result = self.aggregate_results(text, responses)
        return aggregated_result