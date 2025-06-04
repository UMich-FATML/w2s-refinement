import os
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import pickle
import argparse
from prompts_general import *
from utils import *

#python eval.py --experiment 'simple'
#['persona', 'simple']

### Inputs
parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment', type=str)
args = parser.parse_args()
experiment = args.experiment
assert experiment in ['persona', 'simple', 'bias']

device = 'cuda'
eval_model = 'gpt-4o-2024-08-06'

if experiment=='simple':
    from prompts_simple import *
    benchmarks = ['chatgpt_questions_test']
    weak_models = ['falcon','llama2','mistral','gemma'] 
elif experiment=='persona':
    from prompts_persona import *
    persona = 'pirate'
    benchmarks = ['tinyTruthfulQA','tinyAlpacaEval','tinyMMLU']
    weak_models = ['falcon','llama2','mistral','gemma'] 
elif experiment=='bias':
    from prompts_bias import *
    benchmarks = ['bias_content', 'bias_style']
    weak_models = ['weak_gpt'] 
else:
    raise NotImplementedError
    
answers = np.load(f'outputs/answers_benchmarks_{experiment}.npy', allow_pickle=True).item()
scores = {}

for bench in benchmarks:

    scores[bench] = {}
    dic = answers[bench]
    
    ### Loading questions
    questions = CreateQuestionsList(bench)

    ### Eval
    if experiment=='persona': system_prompt = GetEvalSystemPrompt(persona)
    else: system_prompt = GetEvalSystemPrompt()

    for key in dic.keys():
        
        if type(dic[key])==list:
            scores[bench][key] = []
            
            for question,answer in tqdm(zip(questions,dic[key])):
                try:
                    user_prompt = GetEvalUserPrompt(question, answer)
                    prompt = FormatInput(system_prompt, user_prompt, model=eval_model)
                    numbers = GPTEval(prompt, model=eval_model)
                    scores[bench][key].append(numbers)
                except:
                    scores[bench][key].append(None)
        else:
            scores[bench][key] = {}
            
            for key2 in dic[key].keys():
                scores[bench][key][key2] = []
                
                for question,answer in tqdm(zip(questions,dic[key][key2])):
                    try:
                        user_prompt = GetEvalUserPrompt(question, answer)
                        prompt = FormatInput(system_prompt, user_prompt, model=eval_model)
                        numbers = GPTEval(prompt, model=eval_model)
                        scores[bench][key][key2].append(numbers)
                    except:
                        scores[bench][key][key2].append(None)

np.save(f'outputs/scores_benchmarks_{experiment}.npy', scores)