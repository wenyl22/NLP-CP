from tot_generator import LLMGenerator
import csv
import json
import glob
from utils.utils import *
import random
BEAM = 16
BRANCH = 4
csv_fields = ["Step", "Weight", "Content"]
generator = LLMGenerator(num_samples=BRANCH, device = "cuda:1")

def main(question: str, answer: str, name: str, mode: str):
    csv_file_path = f"logs/{name}_{mode}.csv"
    with open(csv_file_path, mode='w') as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
    summary_file_path = f"logs/{name}_{mode}.summary"
    cluster_file_path = f"logs/{name}_{mode}.json"

    question = "Question: " + question + "\n"
    steps_set = [{"content": "", "weight": 1}]
    step_count = 0
    sols = []
    while len(steps_set) > 0:
        step_count += 1
        new_steps_set = []
        variants_set = []
        for steps in steps_set:
            result = generator.evaluate(question, steps["content"], step_count, mode)
            num = result["num_unique_clusters"]
            for i in range(num):
                response = result["clusters"][i]["answer"]
                freq = result["clusters"][i]["frequency"]
                variants = result["clusters"][i]["variants"]
                new_steps_set.append({
                    "content": steps["content"] + response,
                    "weight": steps["weight"] * freq,
                })
                if len(variants) > 1:
                    variants_set.append({
                        "content": steps["content"] + variants[1],
                        "weight": 1,
                    })
                with open(cluster_file_path, mode='a') as cluster_file:
                    json.dump(result["clusters"][i], cluster_file)
                    cluster_file.write("\n")
        new_steps_set.sort(key=lambda x: x["weight"], reverse=True)
        new_steps_set = new_steps_set[:min(len(new_steps_set), BEAM - len(sols))]
        with open(csv_file_path, mode='a') as file:
            writer = csv.DictWriter(file, fieldnames=csv_fields)
            for i in range(len(new_steps_set)):
                writer.writerow({
                    "Step": step_count, 
                    "Weight": new_steps_set[i]["weight"], 
                    "Content": new_steps_set[i]["content"], 
                })
        steps_set = []
        sols.extend(steps for steps in new_steps_set if "\\boxed" in steps["content"])
        steps_set = [steps for steps in new_steps_set if "\\boxed" not in steps["content"]]
        if len(sols) >= BEAM:
            break
        if len(steps_set) + len(sols) < BEAM and len(variants_set) > 0:
            random.shuffle(variants_set)
            steps_set.extend(variants_set[:min(len(variants_set), BEAM - len(sols) - len(steps_set))])
            
    std_ans = get_boxed(answer)
    ans_list = [get_boxed(sol["content"]) for sol in sols]
    with open(summary_file_path, "w") as summary_file:
        summary_file.write("QUESTION: \n" + question + "\n")
        summary_file.write("STD: \n" + std_ans + "\n")
        summary_file.write("ANSWERS: \n")
        for i in range(len(ans_list)):
            summary_file.write(ans_list[i] + "|||")
        summary_file.write("\n")
        for sol in sols:
            summary_file.write("SOLUTION: \n" + sol["content"] + "\n")
        
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
        main(question, answer, name, "llm")
        main(question, answer, name, "naive")