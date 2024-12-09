from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import re

class ProcessRewardModel:
    def __init__(self, model_path = None, device='cpu'):
        if model_path == None:
            model_path = '/nvme1/wyl/caches/hub/models--peiyi9979--math-shepherd-mistral-7b-prm/snapshots/45dc0a3c9ec699b645085c098ed38dc99fba4617/'
        good_token = '+'
        bad_token = '-'
        step_tag = 'ки'

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.candidate_tokens = self.tokenizer.encode(f"{good_token} {bad_token}")[1:]
        self.step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1]
        self.device = device 
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(model_path).eval().to(device)

    def get_step_scores(self, question, output):
        output = re.sub(r'Step (\d+).', r'ки\nStep \1.', output)
        output = re.sub(r'\nки', r' ки', output)
        output = output[2:] + ' ки'
        input_for_prm = f"{question} {output}"
        # count 'ки' in input_for_prm
        # print(f"input_for_prm: {input_for_prm}")
        # print(f"Number of 'ки' in input_for_prm: {input_for_prm.count('ки')}")
        input_id = torch.tensor([self.tokenizer.encode(input_for_prm)]).to(self.device)
        # print(f"input_id: {input_id}")
        # print(f"self.step_tag_id: {self.step_tag_id}")
        # print(f"Occurrences of step_tag_id in input_id: {(input_id == self.step_tag_id).sum()}")        
        logits = self.model(input_id).logits[:, :, self.candidate_tokens]
        scores = logits.softmax(dim=-1)[:, :, 0] 
        step_scores = scores[input_id == self.step_tag_id].detach().cpu()
        # assert(step_scores.size(0) == input_for_prm.count('ки'))
        return step_scores

if __name__ == '__main__':
    model = ProcessRewardModel()

# tokenizer = AutoTokenizer.from_pretrained('/nvme1/wyl/caches/hub/models--peiyi9979--math-shepherd-mistral-7b-prm/snapshots/45dc0a3c9ec699b645085c098ed38dc99fba4617/')
# candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
# step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
# model = AutoModelForCausalLM.from_pretrained('/nvme1/wyl/caches/hub/models--peiyi9979--math-shepherd-mistral-7b-prm/snapshots/45dc0a3c9ec699b645085c098ed38dc99fba4617/').eval()

# question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
# output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки""" # 18 is right
# output2 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки""" # 17 is wrong

# for output in [output1, output2]:
#     input_for_prm = f"{question} {output}"
#     input_id = torch.tensor([tokenizer.encode(input_for_prm)])

#     with torch.no_grad():
#         logits = model(input_id).logits[:,:,candidate_tokens]
#         scores = logits.softmax(dim=-1)[:,:,0] 
#         step_scores = scores[input_id == step_tag_id]
#         print(step_scores)
        
# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240])
