from cot_generator import LLMGenerator
from cluster import LLMCluster
import json
import glob
from utils.utils import *
BEAM = 16
DEVICE = "cuda:1"
generator = LLMGenerator(num_samples=BEAM, device = DEVICE, dir="/nvme1/wyl/caches/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/")
cluster = LLMCluster(device = DEVICE)
def main(question: str, answer: str, name: str, mode: str):
    summary_file_path = f"logss/{name}_{mode}.summary"
    diversity_file_path = f"logss/{name}_{mode}.diversity"
    temperature = float(mode.split("_")[1])
    generator.temperature = temperature
    question = "Question: " + question + "\n"
    sols = generator.evaluate(question)
    ans_list = []
    std_ans = get_boxed(answer)
    for _, sol in enumerate(sols):
        if "\\boxed" not in sol:
            continue
        ans_list.append(get_boxed(sol))
    conclusions = []
    with open(summary_file_path, "w") as summary_file:
        summary_file.write("QUESTION: \n" + question + "\n")
        summary_file.write("STD: \n" + std_ans + "\n")
        summary_file.write("ANSWER: \n")
        for i in range(len(ans_list)):
            summary_file.write(ans_list[i] + "|||")
        summary_file.write("\n")
        for sol in sols:
            summary_file.write("SOLUTION: \n" + sol+ "\n")
            conclusions.extend([A[2:] for A in sol.split("Step ")[1:]])
    print(conclusions)
    clusters = cluster.pair_wise_cluster(question, conclusions)
    with open(diversity_file_path, "w") as diversity_file:
        diversity_file.write(str(len(clusters)))
        for i, cl in enumerate(clusters):
            diversity_file.write(f'\n\n\n\n\n-----------------Cluster {i}---------------\n\n\n\n\n')
            for c in cl:
                diversity_file.write('<ELEMENT>' + c + '\n')
    
        
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
        main(question, answer, name, "cot_1.0")