from generator import LLMGenerator
import csv
BEAM = 3
generator = LLMGenerator(num_samples=BEAM, device = "cuda:1")
csv_file_path = "logs.csv"
csv_fields = ["Step", "Weight", "Content"]
with open(csv_file_path, mode='w') as file:
    writer = csv.DictWriter(file, fieldnames=csv_fields)
    writer.writeheader()

def main(question: str, answer: str):
    question = "Question: " + question + "\n\nStep-by-Step Solution: \n"
    steps_set = [{"content": "", "weight": 1}]
    step_count = 0
    while True:
        step_count += 1
        new_steps_set = []
        is_complete = 0
        for steps in steps_set:
            # generate the step_count-th step
            result = generator.evaluate(question, steps["content"], step_count)
            num = result["num_unique_clusters"]
            for i in range(num):
                response = result["clusters"][i]["answer"]
                freq = result["clusters"][i]["frequency"]
                new_steps_set.append({"content": steps["content"] + f"**Step {step_count}**: " + response + "\n", "weight": steps["weight"] * freq})
                if "complete" in response or "Complete" in response:
                    is_complete += new_steps_set[-1]["weight"]
                else:
                    is_complete -= new_steps_set[-1]["weight"]
        new_steps_set.sort(key=lambda x: x["weight"], reverse=True)
        new_steps_set = new_steps_set[:min(len(new_steps_set), BEAM)]
        # attach the new step to the CSV file
        with open(csv_file_path, mode='a') as file:
            writer = csv.DictWriter(file, fieldnames=csv_fields)
            for i in range(len(new_steps_set)):
                writer.writerow({"Step": step_count, "Weight": new_steps_set[i]["weight"], "Content": new_steps_set[i]["content"]})
        if is_complete > 0:
            break
        steps_set = new_steps_set
if __name__ == '__main__':
    main("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "None")