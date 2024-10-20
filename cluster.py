from prompt.cluster_prompt import *
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
import logging
logging.basicConfig(
    filename='cluster.log',
    level=logging.DEBUG,
    format='%(message)s'
)
class LLMCluster:
    def __init__(self, dir = None, device = "cpu"):
        # use Qwen2.5-7B-Instruct
        if dir == None:
            dir = "/nvme1/wyl/caches/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
        self.model = Qwen2ForCausalLM.from_pretrained(dir).to(device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(dir)
        self.step_count = 0
    def is_same_step(self, content: str, a: str, b: str) -> bool:
        split = content.split("**")[:-1]
        content = "**".join(split)
        prompt = content + f"Possibility 1: **Step {self.step_count}**: " + a
        prompt = prompt + f"Possibility 2: **Step {self.step_count}**: " + b
        prompt += "\nDo possibility 1, 2 get the same conclusion? Response: "
        input_text = (
            f"{cluster_system_prompt}\n"
            # f"{cluster_system_response}\n"
            f"{cluster_user_prompt}"
            f"{cluster_user_response}\n"
            f"{prompt}"
        )
        logging.info(f"input_text: {input_text}")
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs, max_length = inputs.size(-1) + 2)[0][inputs.size(1):].to("cpu")
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        logging.info(f"clustering response: {response}")
        logging.info("--------------------------------")
        return "Same" in response or "same" in response