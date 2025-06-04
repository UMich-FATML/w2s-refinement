import os
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import pickle
import argparse
import torch, gc
from prompts_general import *
from utils import *

#python weak_labels_generation.py --experiment 'simple' 
#['persona', 'simple', 'bias']

### Inputs
parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment', type=str)
args = parser.parse_args()
experiment = args.experiment

assert experiment in ['persona', 'simple', 'bias']

### If we are running the gender bias experiment, we generate biographies for famous women
if experiment=='bias':
    from prompts_bias import *
    model = 'gpt-4o-2024-08-06'
    biographies = {'bio':[], 'question':[], 'corrupted_bio':[]}
    
    for job in tqdm(male_dominated_jobs):
    
            ## Generate bio
            system_prompt = GetBiographySystemPrompt()
            user_prompt =  GetBiographyUserPrompt(job, unbias=True)
            prompt = FormatInput(system_prompt, user_prompt, model)
            biographies['bio'].append(QueryModel(prompt, model, api='OpenAI'))
    
            ## Generate question
            system_prompt = GetQuestionSystemPrompt()
            user_prompt =  GetQuestionUserPrompt(biographies['bio'][-1])
            prompt = FormatInput(system_prompt, user_prompt, model)
            biographies['question'].append(QueryModel(prompt, model, api='OpenAI'))
    
            ## Corrupt bios
            system_prompt = GetFactualErrorsSystemPrompt()
            user_prompt =  GetFactualErrorsUserPrompt(biographies['bio'][-1])
            prompt = FormatInput(system_prompt, user_prompt, model)
            biographies['corrupted_bio'].append(QueryModel(prompt, model, api='OpenAI'))

    biographies['names'] = []
    for bio in biographies['bio']:
        system_prompt = "Return the full name of the person described in the following biography. Please return just the full name, nothing else."
        user_prompt =  bio
        prompt = FormatInput(system_prompt, user_prompt, model)
        biographies['names'].append(QueryModel(prompt, model, api='OpenAI'))
    #biographies['names'] = [bio.split('\n\n')[0].split('Name: ')[1] for bio in tqdm(biographies['corrupted_bio'])]

    np.save(f'outputs/biographies_{experiment}.npy', biographies)

else:
    ### Based on the experiment, we choose the appropriate options
    if experiment=='simple':
        from prompts_simple import *
        models = ['falcon','llama2','mistral','gemma']
        tiny_data = chatgpt_questions_train
        system_prompt = GetWeakModelSystemPrompt()
    elif experiment=='persona':
        from prompts_persona import *
        models = ['falcon','llama2','mistral','gemma']
        DollyData = load_dataset("databricks/databricks-dolly-15k")
        tiny_data = DollyData['train']['instruction'][:100]
        persona = 'pirate'
        system_prompt = GetWeakModelSystemPrompt(persona=persona)
    
    ### Getting answers to the questions
    qas = {'questions':{}, 'answers':{}}
    for model in models:
        qas['questions'][model] = []
        qas['answers'][model] = []
    
        generator = LoadModel(model)
        
        for question in tqdm(tiny_data):
    
            if experiment=='simple':
                user_prompt = GetQuestion(question)
            elif experiment=='persona':
                user_prompt = GetWeakModelUserPrompt(question)
 
            prompt = FormatInput(system_prompt, user_prompt, model)
            weak_resp = QueryModel(prompt, model, generator=generator, device=device)
            qas['questions'][model].append(question)
            qas['answers'][model].append(weak_resp)
                           
        #Flush GPU memory
        del(generator)
        gc.collect()
        torch.cuda.empty_cache()

    ### Saving the answers
    np.save(f'outputs/weak_outputs_{experiment}.npy', qas)