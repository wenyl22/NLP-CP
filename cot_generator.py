from typing import List, Dict
from prompt.cot_generator_prompt import *
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
import torch
class LLMGenerator:
    def __init__(self, dir = None, num_samples = 5, device = "cpu"):
        if dir == None:
            dir = "/nvme1/wyl/caches/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75/"
        self.dir = dir
        self.num_samples = num_samples
        with torch.no_grad():
            self.model = Qwen2ForCausalLM.from_pretrained(dir).to(device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(dir)
        self.temperature = 0.8

    def generate_responses(self, text: str) -> List[str]:
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, do_sample = True, num_return_sequences = self.num_samples, max_new_tokens = 2048, temperature = self.temperature)
        generated_ids = [output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [
            {
                "content": "Step 1. " + response,
                "token_number": len(generated_id),
                "total_token_number": len(generated_id) * self.num_samples
            }
            for response, generated_id in zip(responses, generated_ids)
        ]

    def evaluate(self, question: str):
        messages = [
            {"role": "system", "content": cot_generator_system_prompt},
            {"role": "user", "content": question}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
        text = text + "\nLet's think step by step!\n Step 1. "
        responses = self.generate_responses(text)
        return responses
