import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import pickle
import argparse
from prompts_general import *
from utils import *

#python fine_tune.py --experiment 'simple' 
#['persona', 'simple', 'bias']

### Inputs
parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment', type=str)
args = parser.parse_args()
experiment = args.experiment
assert experiment in ['persona', 'simple', 'bias']

device = 'cuda'
improvement_models = ['gpt-3.5-turbo','gpt-4o-mini-2024-07-18']
frac_train = 1 #Do not modify this. Validation data will not be used anyways...

### Based on the experiment, we choose the appropriate options
if experiment=='simple':
    from prompts_simple import *
    qas = np.load('outputs/weak_outputs_simple.npy', allow_pickle=True).item()
    models = ['falcon','llama2','mistral','gemma'] 
elif experiment=='persona':
    from prompts_persona import *
    qas = np.load('outputs/weak_outputs_persona.npy', allow_pickle=True).item()
    models = ['falcon','llama2','mistral','gemma'] 
elif experiment=='bias':
    from prompts_bias import *
    bios = np.load('outputs/biographies_bias.npy', allow_pickle=True).item()
    models = ['weak_gpt'] 
else:
    raise NotImplementedError
    
qas_improved = np.load(f'outputs/improved_outputs_{experiment}.npy', allow_pickle=True).item()
if experiment in ['persona','simple']:
    qas_icl_improved = np.load(f'outputs/icl_improved_outputs_{experiment}.npy', allow_pickle=True).item()

### Formating the train/val data
system_prompt = GetFineTuneSystemPrompt()

for improvement_model in improvement_models:
    for model in models:
        if experiment in ['persona','simple']:
            questions = qas['questions'][model]
            answers = qas['answers'][model]
            questions_icl_improved = qas_icl_improved['questions'][model+"_"+improvement_model]
            answers_icl_improved = qas_icl_improved['answers'][model+"_"+improvement_model]
        elif experiment=='bias':
            questions = [GetBiographyUserPrompt(job, unbias=False, only_name=False) for job in male_dominated_jobs]
            answers = bios['corrupted_bio']
        questions_improved = qas_improved['questions'][model+"_"+improvement_model]
        answers_improved = qas_improved['answers'][model+"_"+improvement_model]
        
        n_train = int(frac_train*len(questions))
        
        SaveJSONL(system_prompt, questions[:n_train],
                  answers, f'outputs/weak_training_set_{model}_{experiment}_{improvement_model}.jsonl')
        SaveJSONL(system_prompt, questions[n_train:],
                  answers, f'outputs/weak_validation_set_{model}_{experiment}_{improvement_model}.jsonl')
        SaveJSONL(system_prompt, questions_improved[:n_train],
                  answers_improved[:n_train], f'outputs/improved_training_set_{model}_{experiment}_{improvement_model}.jsonl')
        SaveJSONL(system_prompt, questions_improved[n_train:],
                  answers_improved[n_train:], f'outputs/improved_validation_set_{model}_{experiment}_{improvement_model}.jsonl')
        if experiment in ['persona','simple']:
            SaveJSONL(system_prompt, questions_icl_improved[:n_train],
                      answers_icl_improved[:n_train], f'outputs/icl_improved_training_set_{model}_{experiment}_{improvement_model}.jsonl')
            SaveJSONL(system_prompt, questions_icl_improved[n_train:],
                      answers_icl_improved[n_train:], f'outputs/icl_improved_validation_set_{model}_{experiment}_{improvement_model}.jsonl')
    
    for model in list(models):
        print("\n",model)
        CheckTokens(f'outputs/weak_training_set_{model}_{experiment}_{improvement_model}.jsonl', f'outputs/weak_validation_set_{model}_{experiment}_{improvement_model}.jsonl')
        CheckTokens(f'outputs/improved_training_set_{model}_{experiment}_{improvement_model}.jsonl', f'outputs/improved_validation_set_{model}_{experiment}_{improvement_model}.jsonl')
        if experiment in ['persona','simple']:
            CheckTokens(f'outputs/icl_improved_training_set_{model}_{experiment}_{improvement_model}.jsonl', f'outputs/icl_improved_validation_set_{model}_{experiment}_{improvement_model}.jsonl')
    
    ### Finetuning
    fine_tune = {}
    for model in models:
        fine_tune[model] = {'weak':{}, 'improved':{}}
        if experiment in ['persona','simple']:
            fine_tune[model]['icl_improved'] = {}
        for resp in fine_tune[model].keys():
            fine_tune[model][resp]['job_id'] = FineTune(f'outputs/{resp}_training_set_{model}_{experiment}_{improvement_model}.jsonl',
                                                        f'outputs/{resp}_validation_set_{model}_{experiment}_{improvement_model}.jsonl')
        for resp in fine_tune[model].keys():
            fine_tune[model][resp]['model_id'] = GetFineTunedModelName(fine_tune[model][resp]['job_id'])
    
    np.save(f'outputs/fine_tune_{experiment}_{improvement_model}.npy', fine_tune)