from generator import LLMGenerator
import csv
import json
import time
import re
import glob
BEAM = 8
BRANCH = 4
generator = LLMGenerator(num_samples=BRANCH, device = "cuda:3")

def main(question: str, answer: str, name: str, mode: str):

    csv_file_path = f"logs/{name}_{mode}.csv"
    csv_fields = ["Step", "Weight", "Content"]
    with open(csv_file_path, mode='w') as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
    text_file_path = f"logs/{name}_{mode}.log"
    result_file_path = f"logs/{name}_{mode}.txt"
    summary_file_path = f"logs/{name}_{mode}.summary"
    question = "Question: " + question + "\n"
    steps_set = [{"content": "", "weight": 1}]
    step_count = 0
    sols = []
    while len(steps_set) > 0:
        step_count += 1
        new_steps_set = []
        for steps in steps_set:
            result = generator.evaluate(question, steps["content"], step_count, mode)
            num = result["num_unique_clusters"]
            with open(text_file_path, "a") as text_file:
                text_file.write("----------------------------------------------\n")
                for i in range(num):
                    response = result["clusters"][i]["answer"]
                    freq = result["clusters"][i]["frequency"]
                    variants = result["clusters"][i]["variants"]
                    new_steps_set.append({"content": steps["content"] + response, "weight": steps["weight"] * freq})
                    text_file.write(f"***********cluster {i}**********\n")
                    for var in variants:
                        text_file.write(var + "\n")
                    text_file.write(f"********************************\n")
                text_file.write("-----------------------------------------------\n")
        new_steps_set.sort(key=lambda x: x["weight"], reverse=True)
        new_steps_set = new_steps_set[:min(len(new_steps_set), BEAM - len(sols))]
        with open(csv_file_path, mode='a') as file:
            writer = csv.DictWriter(file, fieldnames=csv_fields)
            for i in range(len(new_steps_set)):
                writer.writerow({"Step": step_count, "Weight": new_steps_set[i]["weight"], "Content": new_steps_set[i]["content"]})
        steps_set = []
        for steps in new_steps_set:
            if "\\boxed" in steps["content"]:
                sols.append(steps)
            else:
                steps_set.append(steps)
        if len(sols) >= BEAM:
            break
    ans_list = []
    def extract_boxed_content(text):
        d = 0
        lft = 0
        rgt = 0
        S = text.split("\\boxed")[1]
        for i, c in enumerate(S):
            if c == "{" and d == 0:
                lft = i
            elif c == "}" and d == 1:
                rgt = i
                break
            if c == "{":
                d += 1
            elif c == "}":
                d -= 1
        return S[lft+1:rgt]
    std_ans = extract_boxed_content(answer)
    for sol in sols:
        ans_list.append(extract_boxed_content(sol["content"]))
    with open(summary_file_path, "w") as summary_file:
        summary_file.write("QUESTION: \n" + question + "\n")
        summary_file.write("ANSWER: \n" + std_ans + "\n")
        summary_file.write("SOLUTIONS: \n")
        for i in range(len(ans_list)):
            summary_file.write(ans_list[i] + "|||")

    with open(result_file_path, "w") as result_file:
        result_file.write("QUESTION: \n" + question + "\n")
        result_file.write("ANSWER: \n" + answer + "\n")
        for sol in sols:
            result_file.write("SOLUTION: \n" + sol["content"] + "\n")
        
if __name__ == '__main__':
    num = 100
    files = sorted(glob.glob("data/MATH/test/*/*.json"))
    for file in files:
        field = file.split("/")[-2]
        i = file.split("/")[-1].split(".")[0]
        with open(f"data/MATH/test/{field}/{i}.json") as f:
            data = json.load(f)
            if int(data["level"].split(" ")[-1]) < 3:
                continue
            main(data["problem"], data["solution"], f"MATH_{field}_" + str(i), "naive")
            main(data["problem"], data["solution"], f"MATH_{field}_" + str(i), "llm")
            main(data["problem"], data["solution"], f"MATH_{field}_" + str(i), "llm2")

            num -= 1
        if num == 0:
            break