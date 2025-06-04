from openai import AzureOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import tiktoken
from collections import defaultdict
import numpy as np
import time
from IPython.display import clear_output
import re

try:
    if load_dotenv('env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()
 
os.chdir(os.path.dirname(os.path.abspath(os.getcwd()+"/functions.py")))
um_client = AzureOpenAI(
    api_key=os.environ['API_KEY'],
    api_version="2023-05-15",
    azure_endpoint = os.environ['ENDPOINT_URL'],
    organization = os.environ['ORGANIZATION']
)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
seed=42

def FineTune(training_file_name, validation_file_name=None, model = "gpt-4o-mini-2024-07-18"): #validation_data is not currently used
    
    training_response = openai_client.files.create(
        file=open(training_file_name, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response.id
    
    #validation_response = openai_client.files.create(
    #    file=open(validation_file_name, "rb"), purpose="fine-tune"
    #)
    #validation_file_id = validation_response.id
    
    response = openai_client.fine_tuning.jobs.create(
        training_file=training_file_id,
        #validation_file=validation_file_id,
        model=model, # Enter base model name. Note that in Azure OpenAI the model name contains dashes and cannot contain dot/period characters. 
    )
    
    job_id = response.id

    return job_id


def SaveJSONL(system_prompt, questions, answers, file_name):
    # Create a list to hold our formatted data
    formatted_data = []
    # Iterate over the questions and answers and create the required dictionary structure (Questions and Answers in list form)
    for i in range(len(questions)):
        # Ensure there is a corresponding answer for each question
        if i < len(answers):
            message_entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": questions[i]},
                    {"role": "assistant", "content": answers[i]}
                ]
            }
            formatted_data.append(message_entry)

    with open(file_name, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")



def GetFineTunedModelName(job_id):

    start_time = time.time()
    
    # Get the status of our fine-tuning job.
    response = openai_client.fine_tuning.jobs.retrieve(job_id)
    
    status = response.status
    
    # If the job isn't done yet, poll it every 10 seconds.
    while status not in ["succeeded", "failed"]:
        time.sleep(10)
        response = openai_client.fine_tuning.jobs.retrieve(job_id)
        print(response.model_dump_json(indent=2))
        print("Elapsed time: {} minutes {} seconds".format(int((time.time() - start_time) // 60), int((time.time() - start_time) % 60)))
        status = response.status
        print(f'Status: {status}')
        clear_output(wait=True)
    
    print(f'Fine-tuning job {job_id} finished with status: {status}')
    
    # List all fine-tuning jobs for this resource.
    print('Checking other fine-tune jobs for this resource.')
    response = openai_client.fine_tuning.jobs.list()
    print(f'Found {len(response.data)} fine-tune jobs.')
    
    response = openai_client.fine_tuning.jobs.retrieve(job_id)
    print(response.model_dump_json(indent=2))
    fine_tuned_model = response.fine_tuned_model
    
    return fine_tuned_model

def QueryModel(prompt,
               model,
               model_dict=None,
               max_tokens=2000,
               device='cuda',
               api='UM'):

    if 'gpt' in model: #eg, GPT-3.5 or finetuned GPT-3.5
        
        assert type(prompt)==dict

        if api=='UM':
            client = um_client
        else:
            client = openai_client
            
        #In case prompt breaks Open AI Terms
        try:
            response = client.chat.completions.create(model=model,
                                                  messages=[
                                                      {"role": "system", "content": prompt['system_prompt']},
                                                      {"role": "user", "content": prompt['user_prompt']},
                                                     ],
                                                  temperature=0,
                                                  max_tokens=max_tokens,
                                                  top_p=1,
                                                  seed=seed,
                                                  frequency_penalty=0,
                                                  presence_penalty=0,
                                                  stop=None,
                                                  n=1)
            return response.choices[0].message.content
        except:
            return None
        
    else:
        raise NotImplementedError
        
    return prompt

def FormatInput(system_prompt, user_prompt, model, example=None):

    if example==None:
        if model=='llama2':
            prompt = f"""<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt} [/INST]"""
            
        elif model == 'falcon':
            string = user_prompt
            prompt = f"""##General Rules\n\n{system_prompt}\n\n##{string}"""
            
        elif 'gpt' in model: #eg, GPT-3.5 or finetuned GPT-3.5
            prompt = {'system_prompt':system_prompt, 'user_prompt':user_prompt}   
            
        else:
            raise NotImplementedError

    else:
        if model=='llama2':
            prompt = f"""<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{example['question']} [/INST]\n{example['answer']}\n</s>\n<s>[INST]\n{user_prompt}\n[/INST]"""
            
        elif model == 'falcon':
            string = example['question']+"\n"+example['answer']+"\n\n"+user_prompt
            prompt = f"""##General Rules\n\n{system_prompt}\n\n##Examples\n\n{string}"""
            
        elif 'gpt' in model: #eg, GPT-3.5 or finetuned GPT-3.5
            prompt = {'system_prompt':system_prompt, 'user_prompt':user_prompt}   
            
        else:
            raise NotImplementedError
        
    return prompt

def CheckTokens(training_file_name, validation_file_name=None): #validation_data is not currently used

    #files = [training_file_name, validation_file_name]
    files = [training_file_name]
    
    encoding = tiktoken.get_encoding("cl100k_base") # default encoding used by gpt-4, turbo, and text-embedding-ada-002 models
    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens
    def print_distribution(values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")
        print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")
    
    for file in files:
        print(f"Processing file: {file}")
        with open(file, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        total_tokens = []
        assistant_tokens = []
        for ex in dataset:
            messages = ex.get("messages", {})
            total_tokens.append(num_tokens_from_messages(messages))
            assistant_tokens.append(num_assistant_tokens_from_messages(messages))
        print_distribution(total_tokens, "total tokens")
        print_distribution(assistant_tokens, "assistant tokens")
        print('*' * 50)

def GetEvalSystemPrompt():
    system_prompt = f""" You will be given a mathematical question, a true answer to the question, and a response to the question by an AI assistant. Please act as an impartial grader and evaluate the quality of the response provided by the AI assistant. Your evaluation should consider two primary factors. The first is correctness, the AI response should match the true answer provided. The second is reasoning, the reasoning provided by the AI assistant should match the true answer provided.  If both the answer and reasoning are correct, please provide a score of 1, if either are incorrect please provide a score of 0. For each please strictly following this format: "[[rating]]", for example: "Score: [[1]]". Please do not include anything in your response except the score."""
    return  system_prompt

def GetEvalUserPrompt(question, key, answer):
    processed_answer = answer.replace("Answer:","").strip() #in some cases, the LLM case use 'Answer:' in their response
    user_prompt = f"""Question: {question} \n\n True Answer: {key} \n\n AI Answer: {processed_answer}"""
    return user_prompt

def RetrieveNumbersInBrackets(text):
    """
    This function retrieves all numbers inside double square brackets ([[ ]]) from the provided text.
    
    Args:
    - text (str): The text from which to extract numbers.
    
    Returns:
    - list of int: A list of numbers found inside double square brackets.
    """
    # Using regular expression to find all occurrences of numbers within double square brackets
    numbers = re.findall(r'\[\[([0-9]+)\]\]', text)
    
    # Converting found strings to integers
    numbers = [int(num) for num in numbers]
    
    return numbers
    

def GPTEval(prompt, n=20, model="gpt-4o"):

    assert isinstance(prompt, dict)==True
        
    response = um_client.chat.completions.create(model=model,
                                              messages=[
                                                {"role": "system", "content": prompt['system_prompt']},
                                                {"role": "user", "content": prompt['user_prompt']},
                                              ],
                                              temperature=1,
                                              max_tokens=500,
                                              top_p=1,
                                              seed=seed,
                                              frequency_penalty=0,
                                              presence_penalty=0,
                                              stop=None,
                                              n=n)

    
    scores = [RetrieveNumbersInBrackets(choice.message.content) for choice in response.choices]
    median_size = np.median([len(z) for z in scores]) #we assume that the median size will be the correct size
    scores = np.mean([z for z in scores if len(z)==median_size], axis=0)
    return scores
