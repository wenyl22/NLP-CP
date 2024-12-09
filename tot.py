from tot_generator import LLMGenerator
import json
from utils.utils import *
from utils.prm import ProcessRewardModel
BEAM_WIDTH = 8
BRANCHING_FACTOR = 4
DEVICE1 = "cuda:2"
DEVICE2 = "cuda:3"
generator = LLMGenerator(beam_width=BEAM_WIDTH, branching_factor=BRANCHING_FACTOR, device=DEVICE1)
prm = ProcessRewardModel(device=DEVICE2)
TOKEN_SUM = 0
def main(question: str, answer: str, name: str, mode: str):
    summary_path = f"logs_tot_8_7b/{name}_{mode}.json"
    question = "Question: " + question + "\n"
    steps_set = [{"content": "", "step_scores": []}]
    new_steps_set = []
    sols = []
    step_count = 0
    TOKEN_SUM = 0
    while len(sols) < BEAM_WIDTH:
        new_steps_set = []
        step_count += 1

        for steps in steps_set:
            results, tok = generator.generate(question, steps, step_count, mode)
            TOKEN_SUM += tok
            for res in results:
                if '\\boxed' in res["content"]:
                    sols.append(res)
                else:
                    new_steps_set.append(res)
        for steps in new_steps_set:
            steps["step_scores"] = prm.get_step_scores(question, steps["content"]).detach().cpu().numpy().tolist()

        new_steps_set = sorted(new_steps_set, key=lambda x: x["step_scores"][-1], reverse=True)
        new_steps_set = new_steps_set[:min((BEAM_WIDTH - len(sols)) // BRANCHING_FACTOR + 1, len(new_steps_set))]
        steps_set = new_steps_set
    for steps in sols:
        steps["step_scores"] = prm.get_step_scores(question, steps["content"]).detach().cpu().numpy().tolist()
        steps["token_number"] = len(generator.tokenizer.encode(steps["content"]))
        steps["total_token_number"] = TOKEN_SUM
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
        main(question, answer, name, "tot")