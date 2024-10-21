from generator import LLMGenerator
import csv
import json
import time
import re
import glob
BEAM = 3
generator = LLMGenerator(num_samples=BEAM, device = "cpu")

def main(question: str, answer: str, name: str):
    csv_file_path = f"logs/{name}.csv"
    csv_fields = ["Step", "Weight", "Content"]
    text_file_path = f"logs/{name}.log"
    result_file_path = f"logs/{name}.txt"
    with open(csv_file_path, mode='w') as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
    question = "Question: " + question + "\n"
    steps_set = [{"content": "", "weight": 1}]
    step_count = 0
    while True:
        step_count += 1
        new_steps_set = []
        for steps in steps_set:
            result = generator.evaluate(question, steps["content"], step_count)
            num = result["num_unique_clusters"]
            with open(text_file_path, "a") as text_file:
                text_file.write("----------------------------------------------------------\n")
                for i in range(num):
                    response = result["clusters"][i]["answer"]
                    freq = result["clusters"][i]["frequency"]
                    variants = result["clusters"][i]["variants"]
                    new_steps_set.append({"content": steps["content"] + response, "weight": steps["weight"] * freq})
                    text_file.write(f"****************cluster {i}****************\n")
                    for var in variants:
                        text_file.write(var + "\n")
                    text_file.write(f"*****************************************\n")
                text_file.write("----------------------------------------------------------\n")
        new_steps_set.sort(key=lambda x: x["weight"], reverse=True)
        new_steps_set = new_steps_set[:min(len(new_steps_set), BEAM)]
        with open(csv_file_path, mode='a') as file:
            writer = csv.DictWriter(file, fieldnames=csv_fields)
            for i in range(len(new_steps_set)):
                writer.writerow({"Step": step_count, "Weight": new_steps_set[i]["weight"], "Content": new_steps_set[i]["content"]})
        steps_set = new_steps_set
        if "\\boxed" in steps_set[0]["content"]:
            break
    with open(result_file_path, "w") as result_file:
        result_file.write("QUESTION: \n" + question + "\n")
        result_file.write("ANSWER: \n" + answer + "\n")
        result_file.write("SOL: \n" + steps_set[0]["content"] + "\n")
if __name__ == '__main__':
    num = 10
    files = glob.glob("data/MATH/test/*/*.json")
    for file in files:
        split = file.split("/")
        field = split[-2]
        i = re.search(r"\d+", file).group()
        if data["level"] < 3:
            continue
        with open(f"data/MATH/test/{field}/{i}.json") as f:
            data = json.load(f)
            main(data["problem"], data["solution"], f"MATH_{field}_" + str(i))
            num -= 1
        if num == 0:
            break