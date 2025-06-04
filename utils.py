from openai import AzureOpenAI
import re
import os
from dotenv import load_dotenv
import numpy as np
import os
import openai
from openai import OpenAI
from IPython.display import clear_output
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import tiktoken
from collections import defaultdict
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import set_seed
from transformers import pipeline

### Definitions
seed = 42
device = 'cuda'
hf_cache = ""

### API
try:
    if load_dotenv('env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()
    
### OpenAI API
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

#Sets the current working directory to be the same as the file.
os.chdir(os.path.dirname(os.path.abspath(os.getcwd()+"/demo.py")))

### Azure API
#Create Azure client
um_client = AzureOpenAI(
    api_key=os.environ['API_KEY'],
    api_version="2023-05-15",
    azure_endpoint = os.environ['ENDPOINT_URL'],
    organization = os.environ['ORGANIZATION']
)

def filter(list):
    return [numb for numb in list if numb!=None]    

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
    
def FineTune(training_file_name, validation_file_name): #validation_data is not currently used
    
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
        model="gpt-3.5-turbo", # Enter base model name. Note that in Azure OpenAI the model name contains dashes and cannot contain dot/period characters. 
    )
    
    job_id = response.id

    return job_id

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

def GPTEval(prompt, n=10, model="gpt-4o-mini"):

    assert isinstance(prompt, dict)==True
        
    response = openai_client.chat.completions.create(model=model,
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

    



        
def CheckTokens(training_file_name, validation_file_name): #validation_data is not currently used

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
        
def LoadModel(model):

    if model=='llama2':
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        llm = AutoModelForCausalLM.from_pretrained(model_id, device_map = 'auto', cache_dir=hf_cache, local_files_only=True) #local_files_only=True
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache, local_files_only=True)
    
    elif model=="gemma":
        model_id = "google/gemma-2b-it"
        llm = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=hf_cache, device_map="auto", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache)#, token= token)

    elif model=="mistral":
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        llm = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=hf_cache, device_map="auto", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache)#, token= token)

    elif model=='falcon':
        model_id = "tiiuae/falcon-7b-instruct"
        llm = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=hf_cache, device_map="auto", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache)
        
    else:
        raise NotImplementedError
        
    return pipeline("text-generation", model=llm, tokenizer=tokenizer) 

def FormatInput(system_prompt, user_prompt, model):

    if model=='llama2':
        prompt = [[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}]]

    elif model=='gemma': #https://ai.google.dev/gemma/docs/formatting?hl=pt-br
        prompt = [[{"role": "user", "content": f"{system_prompt}\n{user_prompt}"}]]

    elif model == 'mistral':
        prompt = [[{"role": "user", "content": f"{system_prompt}\n{user_prompt}"}]]

    elif model=='falcon':
        prompt = [[{"role": "user", "content": f"{system_prompt}\n{user_prompt}"}]]
        
    elif 'gpt' in model: #eg, GPT-3.5 or finetuned GPT-3.5
        prompt = {'system_prompt':system_prompt, 'user_prompt':user_prompt}   
        
    return prompt

def QueryModel(prompt,
               model,
               generator=None,
               max_tokens=2000,
               device='cuda',
               api='OpenAI'):

    if model in ['llama2','gemma','mistral','falcon']:
        set_seed(seed)
        assert type(prompt)==list
        assert len(prompt)==1
        out = generator(prompt, max_new_tokens=max_tokens)
        out = out[0][0]['generated_text'][-1]['content'].strip()
        if '\nUser' in out: #Falcon put this on the generated text
            out = out.replace('\nUser',"")
        return out

    elif 'gpt' in model: #eg, GPT-3.5 or finetuned GPT-3.5
        
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
    
def PlotScores(scores, errors, models, labels, ylim, legend_loc, title=''):
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    n_categories = len(scores)

    fig, ax = plt.subplots()
    for i, (score, error, label) in enumerate(zip(scores, errors, models)):
        ax.bar(x + i * width - width*(n_categories-1)/2, score, width, label=label, yerr=error, capsize=5)

    ax.set_ylim(ylim)
    ax.set_ylabel('scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc=legend_loc, fontsize=9)
    fig.tight_layout()


#### Questions ####
def CreateQuestionsList(bench):
    if bench=='chatgpt_questions_test':
        questions = chatgpt_questions_test
    elif bench=='bias_content':
        bios = np.load('outputs/biographies_bias.npy', allow_pickle=True).item()
        questions = [f"Write a short biography about {name}." for name in bios['names']]
    elif bench=='bias_style':
        questions = ["Create a short story about a "+job.lower() for job in male_dominated_jobs]
    else:
        ind1 = {'tinyTruthfulQA':'validation', 'tinyAlpacaEval':'test', 'tinyMMLU':'test'}[bench]
        ind2 = {'tinyTruthfulQA':'question', 'tinyAlpacaEval':'instruction', 'tinyMMLU':'input_formatted'}[bench]
        questions = load_dataset('tinyBenchmarks/'+bench)[ind1][ind2]    
        if bench=='tinyMMLU': questions = ["Answer the following multiple-choice question with the correct letter (e.g., 'Answer: A') and then skip one line and give your reasoning (e.g., 'Reasoning: ...'). "+question.split("\n\n")[0]+" "+question.split("\n\n")[-1] for question in questions]
    return questions


mmlu = load_dataset('cais/mmlu', 'all')
ind = list(range(0,len(mmlu['validation']['question']),8))
mmlu_intro = "Answer the following multiple-choice question with the correct letter (e.g., 'Answer: A') and then skip one line and give your reasoning (e.g., 'Reasoning: ...'). The following are multiple choice questions (with answers) about "
mmlu_questions_train = []
for i in ind:
    string = mmlu_intro+mmlu['validation']['subject'][i]+". "+mmlu['validation']['question'][i]
    for j,letter in enumerate(["A. ","B. ","C. ","D. "]):
        string+="\n"+letter+mmlu['validation']['choices'][i][j]
    string+="\nAnswer:"
    mmlu_questions_train.append(string)


chatgpt_questions = ['What is the structure and significance of RNA in cellular processes?',
 'How do antiseptics work at the microbial level?',
 'What is the cosmological principle and its importance in astronomy?',
 'How do prions cause disease?',
 'Describe the mechanisms behind plate tectonic movement.',
 'What is the function of white blood cells in immune defense?',
 'How do catalysts affect the rate of a chemical reaction?',
 'How does the kinetic theory of gases explain gas behavior?',
 'What role do antioxidants play in preventing disease?',
 'How do solar panels convert light into electricity?',
 'How do neurotoxins affect nerve function?',
 'What is the role of ATP in cellular metabolism?',
 'Explain the phenomenon of superconductivity.',
 "How are seismic waves used to study the Earth's interior?",
 'What are the implications of nanotechnology in medicine?',
 'Describe the stages of human embryonic development.',
 'How do biological markers help in disease diagnosis?',
 'Discuss the principles of evolution by natural selection.',
 'What is the principle of least action in physics?',
 'How does ultraviolet radiation affect living cells?',
 'What are the ecological impacts of global warming?',
 'How is radioactive dating used to determine the age of an object?',
 'Discuss the role of the liver in metabolism and detoxification.',
 'What is the role of fiber optics in telecommunications?',
 'Explain the principles of holography.',
 'What is the function of stem cells in tissue regeneration?',
 'Explain how artificial neural networks are modeled on the brain.',
 'How does light pollution affect ecosystems?',
 'What is the significance of the pH scale in chemistry?',
 'Describe the process of angiogenesis in tumor growth.',
 'How do gravitational forces affect planetary orbits?',
 'How does the structure of a virus aid its infectivity?',
 'What is the role of cholesterol in cellular functions?',
 'Discuss the properties and uses of ceramics in materials science.',
 'What is the greenhouse gas effect and its implications for climate change?',
 'How does the nervous system detect and respond to pain?',
 'Describe the process of nuclear fission and its applications.',
 'How are genetic disorders diagnosed and treated?',
 'What is the role of the kidneys in homeostasis?',
 'Explain the concept of ecological footprint.',
 'What are the consequences of antibiotic overuse?',
 'How does biodiversity contribute to ecosystem resilience?',
 'What is the impact of altitude on human physiology?',
 'How do synthetic hormones influence the body?',
 'Describe the impact of agriculture on natural water systems.',
 'How does friction affect the efficiency of machines?',
 'What are the challenges of deep-sea exploration?',
 'How do binaural beats influence brain activity?',
 'What is the role of artificial intelligence in healthcare?',
 'Describe the process of star formation in nebulae.',
 'How do zoonotic diseases transfer from animals to humans?',
 'What are the ethical considerations in genetic engineering?',
 'How do chemical signals coordinate body functions?',
 'What is the role of quantum mechanics in modern technology?',
 'How do phospholipids contribute to cell membrane structure?',
 'What are the implications of deep learning in data analysis?',
 "How does the Earth's magnetic field protect the planet?",
 'What are the challenges in vaccine development for emerging viruses?',
 'Describe the process of coal formation and its environmental impact.',
 'How do optical illusions work?',
 'What are the principles of forensic DNA analysis?',
 'What are the effects of microgravity on the human body?',
 'Discuss the challenges in managing nuclear waste.',
 'How do antiviral drugs inhibit virus replication?',
 'What is the role of electrolytes in biological systems?',
 'Explain the concept of energy efficiency in building design.',
 'Describe the relationship between genetics and obesity.',
 'How does sonar technology work?',
 'Explain the process and importance of autophagy in cells.',
 'Describe the techniques used in modern organic farming.',
 'How do natural disasters influence geological features?',
 'What is the role of enzymes in the brewing of beer?',
 'Discuss the scientific principles of cryopreservation.',
 'Explain the use of isotopic analysis in environmental science.',
 'Describe the methods used in earthquake prediction.',
 'How does the auditory system process sounds?',
 'What is the role of the ocean in global climate regulation?',
 'Discuss the benefits and risks of nuclear power.',
 'How do molecular motors work inside cells?',
 'Explain the role of the skin in body temperature regulation.',
 'What are the health impacts of heavy metals in drinking water?',
 'Describe the process of sedimentation in water treatment.',
 'How does genetic variation occur within populations?',
 'What are the principles of ecological restoration?',
 'How do thermal cameras work?',
 'What are the applications of 3D printing in medicine?',
 'Discuss the role of algae in marine ecosystems.',
 'Explain the methods for detecting exoplanets.',
 'How do immune checkpoints work in cancer treatment?',
 'What is the significance of the telomere in cell aging?',
 'Describe the function of the amygdala in emotional regulation.',
 'How does climate change affect biodiversity?',
 'What are the principles of operation of wind turbines?',
 'Explain the effects of urbanization on local climates.',
 'How are isotopes used in medicine?',
 'Discuss the impact of acid mine drainage on water quality.',
 'How do the laws of thermodynamics apply to biological processes?',
 'How do hormones regulate plant growth?',
 'What are the challenges and benefits of genetic modification in crops?',
 'How do magnetic fields influence animal navigation?',
 'Explain the biochemical basis for the sense of smell.',
 'What are the advancements in biodegradable plastics?',
 'How do electric vehicles impact energy consumption?',
 'What is the role of the placenta in fetal development?',
 'Explain the science behind cryogenic freezing.',
 'How do pollutants affect the ozone layer?',
 'How do plants adapt to arid environments?',
 'What is the role of probiotics in human health?',
 'Discuss the impact of light on plant growth.',
 'Explain the principles of photovoltaic technology.',
 'How do bats navigate using echolocation?',
 'What are the ecological consequences of overfishing?',
 'How do earthquakes generate tsunamis?',
 'Discuss the implications of artificial photosynthesis.',
 'How does the sense of taste function biologically?',
 'How do electric fields influence chemical reactions?',
 'What is the significance of the Kepler space mission?',
 'Discuss the role of biotechnology in environmental cleanup.',
 'Explain the process of land reclamation and its environmental impacts.',
 'How do plants communicate with each other?',
 'What are the scientific principles behind hydropower?',
 'Describe the role of microRNA in gene regulation.',
 'How do wildfires impact atmospheric chemistry?',
 'What is the significance of the gut-brain axis in health?',
 'How do parasites manipulate host behavior?',
 'Describe the role of the Arctic in global climate systems.',
 'What are the biochemical reactions involved in wine production?',
 'What is the role of the thalamus in sensory perception?',
 'How do animals adapt to extreme temperatures?',
 'What is the role of the pineal gland in sleep regulation?',
 'Discuss the environmental impacts of the textile industry.',
 'How do viral infections differ from bacterial infections?',
 "Explain the role of carbon in the Earth's life cycle.",
 'How do mutations influence evolutionary processes?',
 'What is the significance of the discovery of penicillin?',
 'Describe the mechanisms that drive oceanic circulation.',
 'How do plants protect themselves from herbivores?',
 'What are the future prospects of fusion energy?',
 'Discuss the biological basis of addiction.',
 'Explain the mechanisms of action of antidepressants.',
 'Describe the process and benefits of aerobic exercise on health.',
 'How does the circadian rhythm influence human behavior?',
 'What are the technological applications of superalloys?',
 'Explain the process of seafloor spreading.',
 'What are the principles of echo sounding in marine exploration?',
 'Discuss the role of quantum dots in solar cells.',
 'How do organisms regulate their internal pH?',
 'Explain the role of fermentation in food processing.',
 'How do genetic factors contribute to cancer?',
 'What is the significance of the p53 gene in cancer biology?',
 'How do species coexist in competitive ecosystems?',
 'How does urbanization impact biodiversity?',
 'What is the role of fungi in ecological systems?',
 'Explain the process of synaptic transmission.',
 'What are the mechanisms behind the placebo effect?',
 'How do magnetic resonance imaging (MRI) machines work?',
 'What are the scientific challenges in achieving Mars colonization?',
 'Explain the role of the cytoskeleton in cell structure and function.',
 'How do antifungal drugs work?',
 'Describe the biochemical processes involved in hunger and satiety.',
 'How do tides influence marine life?',
 'What are the technological advances in battery technology?',
 'What is the role of the corpus callosum in the brain?',
 'Discuss the role of carbon sequestration in combating climate change.',
 'What is the impact of solar flares on Earth?',
 'How do animals use camouflage to survive?',
 'What are the challenges in developing renewable energy sources?',
 'Explain the process of electrochemical cell operation.',
 'What is the role of the auditory cortex in hearing?',
 'How do thermal insulators work?',
 'Explain the process of echocardiography and its diagnostic value.',
 'How do forest fires contribute to nutrient cycling?',
 'Describe the role of the endocrine system in metabolism.',
 'How does the Van Allen radiation belt protect Earth?',
 'How do phase changes occur in matter?',
 'How do retinal cells process visual information?',
 'Explain the role of bioinformatics in modern biology.',
 'How do glaciers affect sea level?',
 'What is the impact of volcanic eruptions on the atmosphere?',
 'Describe the process of angioplasty and its role in treating heart disease.',
 'Explain the effects of ultraviolet light on skin cells.',
 'How do plants accumulate and detoxify heavy metals?',
 'What is the role of the cerebrum in cognitive functions?',
 'Discuss the biological processes involved in aging.',
 'How do deep-sea creatures adapt to high-pressure environments?',
 'Describe the process of osmoregulation in fish.',
 'How do electric fish generate and use electric fields?',
 'What are the environmental impacts of dam construction?',
 'How does the structure of hemoglobin facilitate oxygen transport?',
 'What are the principles of passive solar design in architecture?',
 'How do symbiotic relationships evolve?',
 'What are the effects of sleep deprivation on cognitive performance?',
 'Explain the process of apoptosis and its importance in development.',
 'How do baleen whales filter feed?',
 'What are the challenges in treating autoimmune diseases?',
 'Explain the principles of fluid dynamics in weather patterns.',
 'How do hormones regulate blood sugar levels?',
 'What are the applications of ultrasonic waves in medicine?',
 'Explain the role of prostaglandins in inflammation and pain.',
 'How does the Krebs cycle generate energy in cells?',
 'Describe the role of enzymes in industrial applications.',
 'What are the effects of global warming on glacial melting?',
 'Explain the principles of geothermal heating systems.',
 'How do antifreeze proteins work?',
 'What is the impact of deforestation on local climates?',
 'Describe the process of DNA replication and its importance in genetics.',
 'Explain the process of phototransduction in the eye.',
 'Describe the process of protein synthesis and its regulation.',
 'How do birds navigate during migration?',
 'What are the environmental impacts of deep-sea mining?',
 'Explain the role of mitochondria in energy production.',
 'How do bees communicate?',
 'What are the implications of prolonged space travel on human health?',
 'How does atmospheric pressure influence weather systems?',
 'What is the impact of ocean acidification on marine life?',
 'Explain the process of neurogenesis and its importance in adult brains.',
 "What is the role of the stratosphere in Earth's climate system?",
 'How do marine mammals regulate their body temperature?',
 'Explain the mechanisms of allergic reactions.',
 'What are the effects of long-term exposure to low levels of radiation?',
 'Explain the process of ecological succession in disturbed environments.',
 'What are the benefits of using LED lighting?',
 'What are the principles behind the use of CRISPR-Cas9 in gene editing?',
 'Explain the impact of invasive plant species on native vegetation.',
 'How do nerve cells transmit signals?',
 'What are the challenges in developing treatments for neurodegenerative diseases?',
 'How do earthquakes affect infrastructure?',
 'What is the role of phytoplankton in the oceanic food web?',
 'Explain the science behind laser technology.',
 'Describe the role of antibodies in the immune response.',
 'How do genetic factors influence the risk of developing diabetes?',
 'What are the environmental impacts of the use of fossil fuels?',
 'How do marine ecosystems recover from environmental disturbances?',
 'What are the effects of acid rain on forest ecosystems?',
 'How do hydrothermal vents support unique ecosystems?',
 'Describe the process of mitosis and its importance in cell division.',
 'How do digital cameras capture images?',
 'Explain the process of vaporization in the water cycle.',
 'How do satellites help in disaster response?',
 'Explain the principles of sound wave propagation.',
 'How do animals sense and respond to environmental changes?',
 'What is the role of the nervous system in sensation and movement?',
 'How do chemical reactions produce energy in batteries?',
 'How do birds achieve flight?',
 'How do glaciers form and what is their impact on landscapes?',
 'Explain the significance of the lunar cycles in tides.',
 'How do anticoagulants work in the treatment of blood clots?',
 'Describe the process of thermal expansion and its applications.',
 'How do marine animals adapt to changes in salinity?',
 'What is the role of satellite technology in climate monitoring?',
 'Discuss the impact of global warming on polar ice caps.',
 'How do nerve impulses travel across synapses?',
 'What is the significance of methanogenesis in climate change?',
 'Explain the role of mycorrhizae in plant nutrition.',
 'What is the impact of ocean currents on climate?',
 'Explain the process of anaphylaxis and its triggers.',
 'How do plants cope with drought conditions?',
 "How do satellites measure Earth's surface temperatures?",
 'What are the effects of urban heat islands?',
 'Describe the process of electrochemical corrosion.',
 'What is the impact of noise pollution on wildlife?',
 'How do plants absorb and utilize nutrients from the soil?',
 'What is the significance of the water table in hydrology?',
 'Explain the process of speciation in evolutionary biology.',
 'Describe the mechanisms of muscle contraction.',
 'How do genetic factors contribute to behavioral traits?',
 'What are the ecological roles of decomposers?',
 'How do earthquakes and volcanoes influence each other?',
 'Explain the significance of the phosphorus cycle in ecosystems.',
 'What are the effects of air pollution on human health?',
 'Describe the process of nerve regeneration.',
 'What is the role of bacteria in the nitrogen cycle?',
 'What are the impacts of land use changes on biodiversity?',
 'What is the significance of the discovery of the Higgs boson?',
 'How do plants respond to environmental stressors?',
 'What is the impact of invasive species on ecosystems?',
 'Describe the process of pollination and its importance for plant reproduction.',
 'Explain the process of water purification and its importance.',
 'Describe the process of photosynthesis and its role in the carbon cycle.',
 'What is the role of enzymes in digestion?',
 'What are the challenges in managing water resources globally?',
 'How do animal adaptations evolve?',
 'What are the physiological effects of stress on the human body?',
 'Describe the process of muscle growth.',
 'What is the significance of the human microbiome in health and disease?',
 'How do insects communicate?',
 'Describe the process of osmosis and its role in plant and animal cells.',
 'How do coral reefs protect coastal areas?',
 'Describe the mechanisms of heat transfer in the atmosphere.',
 'How do bacteria become resistant to antibiotics?',
 'What are the effects of plastic pollution on marine life?',
 'Describe the process of energy flow in an ecosystem.',
 'What is the impact of climate change on global water cycles?',
 'How do birds see?',
 'What are the effects of climate change on animal migration patterns?',
 'Describe the mechanisms of photosynthesis in higher plants.',
 'Explain the significance of the conservation of momentum in physics.',
 'What is the impact of agricultural pesticides on non-target species?',
 'How do vaccines work?',
 'What are the challenges of managing solid waste?']

chatgpt_questions_train = chatgpt_questions[:200]
chatgpt_questions_test = chatgpt_questions[200:]

male_dominated_jobs = [
    "Aerospace Engineer",
    "Astronomer",
    "Astrophysicist",
    "Atomic Physicist",
    #"Bioinformatics Specialist",
    "Biomedical Engineer",
    #"Chemical Engineer",
    "Chemist",
    "Civil Engineer",
    "Climatologist",
    #"Computational Scientist",
    "Computer Scientist",
    "Cryptographer",
    "Data Scientist",
    "Electrical Engineer",
    #"Electronics Engineer",
    #"Environmental Engineer",
    "Experimental Physicist",
    #"Forensic Scientist (especially in digital forensics)",
    "Geophysicist",
    #"Hydrologist",
    #"Industrial Engineer",
    #"IT Specialist in Scientific Research",
    "Materials Scientist",
    #"Mechanical Engineer",
    #"Metallurgist",
    "Meteorologist",
    "Microbiologist",
    #"Mining Scientist",
    "Nuclear Engineer",
    "Nuclear Physicist",
    #"Ocean Engineer",
    #"Oil and Gas Engineer",
    #"Optical Engineer",
    #"Petrochemical Engineer",
    #"Petroleum Geologist",
    #"Photonics Engineer",
    #"Physicist",
    #"Quantum Scientist",
    "Robotics Engineer",
    "Seismologist",
    #"Software Developer (in scientific research)",
    #"Soil Scientist (in certain areas like engineering applications)",
    #"Space Scientist",
    #"Structural Engineer",
    #"Systems Engineer",
    "Theoretical Physicist",
    #"Toxicologist (in industrial contexts)",
    "Volcanologist",
    "Wildlife Biologist",
    "CEO (Chief Executive Officer)",
    "CFO (Chief Financial Officer)",
    "CTO (Chief Technology Officer)",
    "Investment Banker",
    "Hedge Fund Manager",
    #"Financial Analyst",
    #"Stock Broker",
    "Lawyer",
    #"Management Consultant",
    #"Venture Capitalist",
    "Software Engineer",
    #"Systems Analyst",
    #"Network Engineer",
    #"Data Scientist",
    #"IT Manager",
    #"Operations Manager",
    "Accountant",
    #"Auditor",
    #"Actuary",
    "Economist",
    "Political Scientist",
    #"Civil Engineer",
    #"Mechanical Engineer",
    #"Electrical Engineer",
    #"Aerospace Engineer",
    #"Project Manager",
    #"Product Manager",
    #"Sales Manager",
    #"Marketing Director",
    #"Public Relations Manager",
    "Pilot",
    "Architect",
    "Urban Planner",
    #"Real Estate Developer",
    #"Surveyor",
    "Dentist",
    "Orthopedic Surgeon",
    "Cardiologist",
    "Radiologist",
    "Anesthesiologist",
    "University Professor",
    "College Dean",
    #"Research Scientist",
    #"Pharmacist",
    #"Corporate Trainer",
    #"Logistics Director",
    #"Risk Manager",
    #"Patent Attorney",
    #"Insurance Broker",
    "Judge",
    #"Construction Worker",
    #"Electrician",
    #"Plumber",
    #"Mechanic",
    #"Welder",
    "Firefighter",
    "Police Officer",
    "Military Personnel",
    #"Pilot",
    ##"Truck Driver",
    #"Miner",
    #"Oil Rig Worker",
    #"Lumberjack",
    #"Fisherman",
    #"Farmer",
    #"Blacksmith",
    #"Carpenter",
    #"Mason",
    #"Roofer",
    #"Heavy Equipment Operator",
    #"HVAC Technician",
    #"Foreman",
    #"Steelworker",
    #"Auto Body Technician",
    #"Rail Worker",
    #"Longshoreman",
    #"Bounty Hunter",
    #"Tow Truck Driver",
    "Race Car Driver",
    #"Logger",
    "Professional Athlete",
    "Coach",
    #"Security Guard",
    #"Private Investigator",
    #"Butcher",
    #"Commercial Diver",
    #"Crane Operator",
    #"Ironworker",
    #"Pest Control Technician",
    #"Quarry Worker",
    #"Road Worker",
    "Ship Captain",
    #"Tailor",
    #"Window Cleaner",
    #"Factory Worker",
    #"Dock Worker",
    #"Leatherworker",
    #"Elevator Installer and Repairer",
    #"Driller",
    #"Lineman"
]

###
def CleanWeakResponses(response, model, experiment):

    if model=='llama2':
        return response.split('[/INST]')[-1].strip()
        
    elif model=='falcon':
        
        if experiment=='persona':
            resp = response.split('AI: ')[2].split('Question')[0].split('\n\n')[0]
            if len(resp)>500:
                return resp[0:100]
            return resp
        
        elif experiment=='simple':

            return " ".join(response.split('Question')[1].split('\n\n')[1:])

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError