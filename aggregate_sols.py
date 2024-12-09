import glob
from math_equivalence import is_equiv
from utils.diversity import *
import os
import json
from utils.utils import get_boxed
from transformers import Qwen2Tokenizer, Qwen2ForSequenceClassification
my_tokenizer = Qwen2Tokenizer.from_pretrained("/nvme1/wyl/caches/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35/")

import random

BUDGET = 8

def main2(file):
    field = file.split('/')[-1].split('_')[1:-2]
    field = '_'.join(field)
    num = file.split('_')[-2]
    # get std
    with open(f'./data/MATH/test/{field}/{num}.json', 'r') as f:
        data = json.load(f)
        level = int(data['level'][-1])
        sol = data['solution']
    # if level <= 3:
    #     file = file.replace('tot_16', 'tot_8')
    # file = file.replace('tot_16', 'cot')
    # file = file.replace('tot', 'cot')

    with open(file, 'r') as f:
        data = json.load(f)
    if 'cot' in file:
    # shuffle the data
        random.shuffle(data)
        data = data[:BUDGET]
    else:
        pass
        # data = sorted(data, key=lambda x: x['step_scores'][-1], reverse=True)
        # data = data[:BUDGET]
    ans_list = []
    token = 0
    step_number = 0
    for d in data:
        token += d["token_number"]
        step_number += d['content'].count('Step')
        if '\\boxed' not in d['content']:
            continue
        is_new = True
        for a in ans_list:
            if is_equiv(get_boxed(d['content']), a['content']):
                a['count'] += 1
                a['scores'] += d['step_scores'][-1]
                is_new = False
        if is_new:
            ans_list.append({'content': get_boxed(d['content']), 'count': 1, 'scores': d['step_scores'][-1]})
    # sort the list by the count of each answer
    #token = d["total_token_number"]
    ans_list = sorted(ans_list, key=lambda x: x['scores'], reverse=True)
    step_number = step_number / len(data)
    if len(ans_list) > 0 and is_equiv(get_boxed(sol), ans_list[0]['content']):
        return level, 1, len(ans_list), token, step_number
    else:
        return level, 0, len(ans_list), token, step_number
        
    
if __name__ == '__main__':
    BUDGET = 16
    files = glob.glob('./logs_tot_16/*')
    L = len(files)
    print(L)
    # files = [file.replace('tot_16', 'tot_8').replace('tot', 'tot') for file in files]
    ACC_TOT = []
    DIV_TOT = []
    S = [0, 0, 0, 0, 0]
    D = [0, 0, 0, 0, 0]
    C = [0, 0, 0, 0, 0]
    T = [0, 0, 0, 0, 0]
    ST = [0, 0, 0, 0, 0]
    for file in files:
        level, acc, div, tok, st = main2(file)
        level -= 1
        S[level] += acc
        D[level] += div
        C[level] += 1
        T[level] += tok
        ST[level] += st
    print([x/y for x, y in zip(S, C)])
    print([x/y for x, y in zip(D, C)])
    print(sum(S)/sum(C))
    print(sum(D)/sum(C))
    print(sum(T)/sum(C))
    print(sum(ST)/sum(C))