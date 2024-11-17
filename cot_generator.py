from typing import List, Dict
from prompt.cot_generator_prompt import *
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
class LLMGenerator:
    def __init__(self, dir = None, num_samples = 5, device = "cpu"):
        if dir == None:
            dir = "/nvme1/wyl/caches/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35/"
        self.dir = dir
        self.num_samples = num_samples
        self.model = Qwen2ForCausalLM.from_pretrained(dir).to(device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(dir)
        self.temperature = 1.0

    def generate_responses(self, text: str) -> List[str]:
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        if "Instruct" in self.dir:
            generated_ids = self.model.generate(**model_inputs, do_sample = True, num_return_sequences = self.num_samples, max_new_tokens = 2048, temperature = self.temperature)
        else:
            # add repetition penalty
            generated_ids = self.model.generate(**model_inputs, do_sample = True, num_return_sequences = self.num_samples, max_new_tokens = 2048, temperature = self.temperature, repetition_penalty = 1.5)
        generated_ids = [output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return responses

    def evaluate(self, question: str):
        messages = [
            {"role": "system", "content": cot_generator_system_prompt},
            {"role": "user", "content": question}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
        text = text + "\nThe step-by-step solution is as follows:\n Step 1. "
        responses = self.generate_responses(text)
        responses = ["Step 1. " + response for response in responses]
        return responses
