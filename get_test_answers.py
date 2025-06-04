import os
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import pickle
import argparse
import torch, gc
from utils import *
from prompts_general import *

#python get_test_answers.py --experiment 'simple' 
#['persona', 'simple']

### Inputs
parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment', type=str)
args = parser.parse_args()
experiment = args.experiment
assert experiment in ['persona', 'simple', 'bias']

device = 'cuda'
improvement_models = ['gpt-3.5-turbo','gpt-4o-mini-2024-07-18']

### Based on the experiment, we choose the appropriate options
if experiment=='simple':
    from prompts_simple import *
    benchmarks = ['chatgpt_questions_test']
elif experiment=='persona':
    from prompts_persona import *
    persona = 'pirate'
    benchmarks = ['tinyTruthfulQA','tinyAlpacaEval','tinyMMLU']
elif experiment=='bias':
    from prompts_bias import *
    benchmarks = ['bias_content', 'bias_style']
else:
    raise NotImplementedError

### Answer
answers = {}

for bench in benchmarks:

    answers[bench] = {}
    
    ### Loading questions
    questions = CreateQuestionsList(bench)

    ### Getting responses
    
    # Weak models
    fine_tune = np.load(f'outputs/fine_tune_{experiment}_{improvement_models[0]}.npy', allow_pickle=True).item()
    weak_models = fine_tune.keys()

    #if False:
    if experiment in ['simple','persona']: # for other experiments, we do not have a weak model
        for weak_model in weak_models:
            answers[bench][weak_model] = []
            model_dict = LoadModel(weak_model)
            for question in tqdm(questions):
                
                if experiment=='simple': #for weak models, we require extra formatting for the user prompt
                    user_prompt = GetQuestion(question)
                elif experiment=='persona':
                    user_prompt = GetWeakModelUserPrompt(question)

                system_prompt = GetWeakModelSystemPrompt()
                prompt = FormatInput(system_prompt, user_prompt, weak_model)
                weak_resp = QueryModel(prompt, weak_model, generator=model_dict, device=device)
                answers[bench][weak_model].append(weak_resp)  

            #Flush GPU memory
            del(model_dict)
            gc.collect()
            torch.cuda.empty_cache()
                    
    # Naive GPTs
    for improvement_model in improvement_models:
        answers[bench]['naive_gpt'+"_"+improvement_model] = []
        for question in tqdm(questions):
            system_prompt = GetFineTuneSystemPrompt()
            user_prompt = question
            prompt = FormatInput(system_prompt, user_prompt, model=improvement_model)
            answers[bench]['naive_gpt'+"_"+improvement_model].append(QueryModel(prompt, model=improvement_model, api='OpenAI'))
            
    # Finetuned GPTs
    for improvement_model in improvement_models:
        fine_tune = np.load(f'outputs/fine_tune_{experiment}_{improvement_model}.npy', allow_pickle=True).item()
        weak_models = fine_tune.keys()
        
        for weak_model in weak_models:
            answers[bench][weak_model+"_"+improvement_model] = {}
    
            for finetuned_model in fine_tune[weak_model].keys():
                answers[bench][weak_model+"_"+improvement_model][finetuned_model] = []
                
                for question in tqdm(questions):
                    model = fine_tune[weak_model][finetuned_model]['model_id']
                    system_prompt = GetFineTuneSystemPrompt()
                    user_prompt = question
                    prompt = FormatInput(system_prompt, user_prompt, model=improvement_model)
                    answers[bench][weak_model+"_"+improvement_model][finetuned_model].append(QueryModel(prompt, model=model, api='OPENAI'))
    
                
np.save(f'outputs/answers_benchmarks_{experiment}.npy', answers)