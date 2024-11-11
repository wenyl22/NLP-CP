from cot_generator import LLMGenerator
import csv
import json
import time
import re
import glob
BEAM = 8
generator = LLMGenerator(num_samples=BEAM, device = "cuda:0")
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

def main(question: str, answer: str, name: str, mode: str):
    result_file_path = f"logs/{name}_{mode}.txt"
    summary_file_path = f"logs/{name}_{mode}.summary"
    question = "Question: " + question + "\n"
    sols = generator.evaluate(question)
    ans_list = []
    std_ans = extract_boxed_content(answer)
    for _, sol in enumerate(sols):
        if "\\boxed" not in sol:
            continue
        ans_list.append(extract_boxed_content(sol))
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
            result_file.write("SOLUTION: \n" + sol+ "\n")
        
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
            main(data["problem"], data["solution"], f"MATH_{field}_" + str(i), "cot")
            num -= 1
        if num == 0:
            break