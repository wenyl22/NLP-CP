def get_inputs(tokenizer, sys_prompt, user_prompt, start = ""):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += start
    model_inputs = tokenizer([text], return_tensors="pt")
    return model_inputs

def get_response(tokenizer, model_inputs, generated_ids):
    generated_ids = [
        output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return response

def get_clusters(response, responses):
    clusters = []
    for cluster in response.split("**[")[1:]:
        cluster = cluster.split("]**")[0]
        def check_validity(s):
            if len(s) == 0:
                return False
            s = s.split(",")
            for i in s:
                if sum(x.isdigit() for x in i) == 0:
                    return False
                if sum(x.isdigit() or x == " " or x == "\n" for x in i) != len(i):
                    return False
            return True
        if not check_validity(cluster):
            continue
        cluster = list(map(int, cluster.split(",")))
        lst = []
        for i in range(len(cluster)):
            if cluster[i] <= len(responses):
                lst.append(responses[cluster[i] - 1])
        if len(lst) > 0:
            clusters.append(lst)
    if len(clusters) == 0:
        clusters.append(responses)
    return clusters

def get_boxed(text):
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