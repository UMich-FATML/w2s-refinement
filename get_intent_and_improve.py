import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse
import pickle
import time
from prompts_general import *
from utils import *

#python get_intent_and_improve.py --experiment 'simple' 
#['persona', 'simple']

### Inputs
parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment', type=str)
args = parser.parse_args()
experiment = args.experiment
assert experiment in ['persona', 'simple', 'bias']

### Based on the experiment, we choose the appropriate options
device = 'cuda'
improvement_models = ['gpt-3.5-turbo','gpt-4o-mini-2024-07-18']

if experiment=='simple':
    from prompts_simple import *
    qas_weak = np.load('outputs/weak_outputs_simple.npy', allow_pickle=True).item()
    models = ['falcon','llama2','mistral','gemma'] 
    batch_size = 4
elif experiment=='persona':
    from prompts_persona import *
    qas_weak = np.load('outputs/weak_outputs_persona.npy', allow_pickle=True).item()
    models = ['falcon','llama2','mistral','gemma']
    batch_size = 5
elif experiment=='bias':
    from prompts_bias import *
    bios = np.load('outputs/biographies_bias.npy', allow_pickle=True).item()
    models = ['weak_gpt']
    batch_size = 13
else:
    raise NotImplementedError

### Inferring intent
intents = {}
system_prompt = GetIntentSystemPrompt()

for model in models:
    for improvement_model in improvement_models:
        if experiment=='bias':
            its = len(bios['bio'])/batch_size
        else:
            its = np.ceil(len(qas_weak['questions'][model])/batch_size) #this will be the same for all models
    
        intents[model+"_"+improvement_model] = []
        
        for it in tqdm(range(int(its))):
            if experiment in ['persona','simple']:
                questions = [q.replace("\n","") for q in qas_weak['questions'][model][(it*batch_size):((it+1)*batch_size)]] # Questions can be model dependent. For example, this will happen if one of the models gives an invalid response for one of the original questions.
                answers = [q.replace("\n","") for q in qas_weak['answers'][model][(it*batch_size):((it+1)*batch_size)]]
            elif experiment=='bias': #this will not be used
                questions = [GetBiographyUserPrompt(job, unbias=False, only_name=False) for job in male_dominated_jobs][(it*batch_size):((it+1)*batch_size)]
                answers = bios['corrupted_bio'][(it*batch_size):((it+1)*batch_size)]
            else:
                raise NotImplementedError
                
            user_prompt =  GetIntentUserPrompt(questions, answers)
            prompt = FormatInput(system_prompt, user_prompt, improvement_model)
            intents[model+"_"+improvement_model].append(QueryModel(prompt, improvement_model, api='OpenAI'))

np.save(f'outputs/intents_{experiment}.npy', intents)

### Generate responses from naive GPT, improved models, and from ICL
naive_answers = {}
improved_answers = {}
icl_improved_answers = {}
for model in models:
    for improvement_model in improvement_models:
        improved_answers[model+"_"+improvement_model] = []
        icl_improved_answers[model+"_"+improvement_model] = []

for improvement_model in improvement_models:
    naive_answers[improvement_model] = []
    count = 0
    
    for it in tqdm(range(int(its))):
        if experiment in ['persona','simple']:
            questions = [q.replace("\n","") for q in qas_weak['questions'][model][(it*batch_size):((it+1)*batch_size)]] # Questions can be model dependent. For example, this will happen if one of the models gives an invalid response for one of the original questions.
        elif experiment=='bias':
            questions = [GetBiographyUserPrompt(job, unbias=False, only_name=False) for job in male_dominated_jobs][(it*batch_size):((it+1)*batch_size)]
        else:
            raise NotImplementedError
            
        for question in questions:
    
            if experiment in ['persona','simple']:
                user_prompt =  GetQuestion(question)
            elif experiment=='bias':
                user_prompt =  question
            else:
                raise NotImplementedError
    
            # Generating naive GPT responses
            system_prompt = GetImprovementSystemPrompt(None)
            prompt = FormatInput(system_prompt, user_prompt, improvement_model)
            naive_answers[improvement_model].append(QueryModel(prompt, improvement_model, api='OpenAI'))
    
            # Generating improved responses
            for model in models:
    
                if experiment in ['persona','simple']:
                    # Improvement with inferred intent
                    intent = intents[model+"_"+improvement_model][it]
                    system_prompt = GetImprovementSystemPrompt(intent)
                    prompt = FormatInput(system_prompt, user_prompt, improvement_model)
                    improved_answers[model+"_"+improvement_model].append(QueryModel(prompt, improvement_model, api='OpenAI'))
    
                    # Improvement with ICL
                    icl_indices = [i for i,q in enumerate(questions) if q!=question]
                    icl_questions = [questions[i] for i in icl_indices]
                    icl_answers = [q.replace("\n","") for q in qas_weak['answers'][model][(it*batch_size):((it+1)*batch_size)]]
                    icl_answers = [icl_answers[i] for i in icl_indices]
                    system_prompt = GetICLUserPrompt(icl_questions, icl_answers)
                    prompt = FormatInput(system_prompt, user_prompt+"\nAnswer:", improvement_model)
                    icl_improved_answers[model+"_"+improvement_model].append(QueryModel(prompt, improvement_model, api='OpenAI'))
                    
                elif experiment=='bias':
                    # Basic improvement (just ask it to improve)
                    system_prompt = GetImprovementSystemPrompt(True)
                    prompt = FormatInput(system_prompt, GetIntentUserPrompt([question], [bios['corrupted_bio'][count]]), improvement_model)
                    improved_answers[model+"_"+improvement_model].append(QueryModel(prompt, improvement_model, api='OpenAI'))
    
                else:
                    raise NotImplementedError
            count+=1
                
### Formating generation
qas_improved = {'questions':{}, 'answers':{}}
if experiment in ['persona','simple']: 
    qas_icl_improved = {'questions':{}, 'answers':{}}

for model in models:
    for improvement_model in improvement_models:
        if experiment in ['persona','simple']:
            questions = qas_weak['questions'][model]
            qas_icl_improved['questions'][model+"_"+improvement_model] = questions
            qas_icl_improved['answers'][model+"_"+improvement_model] = icl_improved_answers[model+"_"+improvement_model]
        elif experiment=='bias':
            questions = [GetBiographyUserPrompt(job, unbias=False, only_name=False) for job in male_dominated_jobs]
        qas_improved['questions'][model+"_"+improvement_model] = questions
        qas_improved['answers'][model+"_"+improvement_model] = improved_answers[model+"_"+improvement_model]
            
np.save(f'outputs/naive_gpt_outputs_{experiment}.npy', naive_answers)
np.save(f'outputs/improved_outputs_{experiment}.npy', qas_improved)
if experiment in ['persona','simple']: 
    np.save(f'outputs/icl_improved_outputs_{experiment}.npy', qas_icl_improved)