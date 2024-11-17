from typing import List, Dict
from cluster import LLMCluster, NaiveCluster
# from prompt.tot_generator_prompt import *
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, StoppingCriteria, StoppingCriteriaList
from utils.utils import *
from prompt.cot_generator_prompt import *
class KeyWordOne_StoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.keyword = None
    def __call__(self,input_ids,scores,**kwargs):
        if len(input_ids[0]) < len(self.keyword):
            return False
        if input_ids[0][len(input_ids[0] - len(self.keyword)):].equal(self.keyword):
            return True
        else:
            return False

class LLMGenerator:
    def __init__(self, dir = None, num_samples = 5, device = "cuda:1"):
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
        self.device = device
        self.stopping_criteria = KeyWordOne_StoppingCriteria()

    def generate_responses(self, model_inputs):
        keyword = self.tokenizer.encode(f"Step {self.step_count + 1}.",add_special_tokens = False,return_tensors = 'pt').squeeze().to(self.device)
        self.stopping_criteria.keyword = keyword
        stopping_criteria_list = StoppingCriteriaList([self.stopping_criteria])
        generated_ids = self.model.generate(
            **model_inputs, 
            max_new_tokens=512, 
            do_sample = True, 
            num_return_sequences = self.num_samples,
            stopping_criteria = stopping_criteria_list
        )
        responses = get_response(self.tokenizer, model_inputs, generated_ids)
        for _, response in enumerate(responses):
            response = response.split(f"Step {self.step_count + 1}. ")[0]
            #if "\\boxed" not in response:
            #    response = self.cluster["llm"].extract(response)
            response = f"Step {self.step_count}. " + response
            responses[_] = response
        return responses

    def aggregate_results(self, context:str, responses: List[str]) -> Dict[str, any]:
        final_answers = responses
        clusters = self.cluster[self.mode].cluster(context, final_answers)
        cluster_info = []
        for cluster in clusters:
            cluster_info.append({
                "answer": cluster[0],
                "frequency": len(cluster),
                "variants": cluster
            })      
        cluster_info.sort(key=lambda x: x['frequency'], reverse=True)
        return {
            "clusters": cluster_info,
            "total_responses": len(responses),
            "num_unique_clusters": len(clusters)
        }

    def evaluate(self, question: str, steps: str, step_count: int, mode: str) -> Dict[str, any]:
        self.mode = mode
        self.step_count = step_count
        self.cluster["llm"].step_count = step_count
        model_inputs = get_inputs(self.tokenizer, cot_generator_system_prompt, question, "\nThe step-by-step solution is as follows:\n" + steps + f"Step {step_count}. ").to(self.device)
        responses = self.generate_responses(model_inputs)
        context = question + "\nThe step-by-step solution is as follows:\n" + steps
        aggregated_result = self.aggregate_results(context, responses)
        return aggregated_result