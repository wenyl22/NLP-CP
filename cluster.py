from prompt.cluster_prompt import *
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
import time

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