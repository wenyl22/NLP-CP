from typing import List, Dict
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
    def __init__(self, dir = None, beam_width = 16, branching_factor = 4, device = "cuda:1"):
        if dir == None:
            dir = "/nvme1/wyl/caches/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75/"
        self.beam_width = beam_width
        self.branching_factor = branching_factor
        self.step_count = 0
        self.model = Qwen2ForCausalLM.from_pretrained(dir).to(device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(dir)
        self.device = device
        self.stopping_criteria = KeyWordOne_StoppingCriteria()
        self.temperature = 0.8

    def generate_responses(self, model_inputs):
        keyword = self.tokenizer.encode(f"Step {self.step_count + 1}",add_special_tokens = False,return_tensors = 'pt').squeeze().to(self.device)
        self.stopping_criteria.keyword = keyword
        stopping_criteria_list = StoppingCriteriaList([self.stopping_criteria])
        generated_ids = self.model.generate(
            **model_inputs, 
            max_new_tokens=2048, 
            do_sample = True, 
            num_return_sequences = self.branching_factor,
            temperature = self.temperature,
            stopping_criteria = stopping_criteria_list
        )
        responses = get_response(self.tokenizer, model_inputs, generated_ids)
        TOKEN_SUM = 0
        for _, response in enumerate(responses):
            response = response.split(f"Step {self.step_count + 1}")[0]
            response = f"Step {self.step_count}. " + response
            TOKEN_SUM += len(self.tokenizer.encode(response))
            responses[_] = {
                "content": response,
            }
        return responses, TOKEN_SUM

    def generate(self, question, steps, step_count, mode: str) -> Dict[str, any]:
        self.mode = mode
        self.step_count = step_count
        model_inputs = get_inputs(self.tokenizer, cot_generator_system_prompt, question, "\nLet's think step-by-step!\n" + steps["content"] + f"Step {step_count}. ").to(self.device)
        responses, TOKEN_SUM = self.generate_responses(model_inputs)
        for _, response in enumerate(responses):
           response["content"] =  steps["content"] + "\n" + response["content"]
        return responses, TOKEN_SUM