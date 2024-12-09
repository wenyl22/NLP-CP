from cot_generator import LLMGenerator
from cluster import LLMCluster
import json
from utils.utils import *
from utils.prm import ProcessRewardModel
BEAM = 64
DEVICE1 = "cuda:2"
DEVICE2 = "cuda:3"
generator = LLMGenerator(num_samples=BEAM, device = DEVICE1)
prm = ProcessRewardModel(device=DEVICE2)
def main(question: str, answer: str, name: str, mode: str):
    summary_path = f"logs_cot_7b/{name}_{mode}.json"
    question = "Question: " + question + "\n"
    sols = generator.evaluate(question)
    # for each solution, record
    # 1. token number
    # 2. content
    # 3. value given by reward model
    for i, sol in enumerate(sols):
        step_scores = prm.get_step_scores(question, sol["content"])
        sols[i]["step_scores"] = step_scores.tolist()
    with open(summary_path, "w") as file:
        json.dump(sols, file)

        
if __name__ == '__main__':
    with open("./data/math500.jsonl", "r") as file:
        data = file.readlines()
    for i in range(len(data)):
        data[i] = json.loads(data[i])
    for i in range(len(data)):
        unique_id = data[i]["unique_id"]
        field = unique_id.split("/")[1]
        num = unique_id.split("/")[2].split(".")[0]
        question = data[i]["problem"]
        answer = data[i]["solution"]
        name = f"MATH_{field}_{num}"
        main(question, answer, name, "cot")
