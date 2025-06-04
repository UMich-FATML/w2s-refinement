def GetWeakModelSystemPrompt(persona='pirate'):

    system_prompt = f"""You are an AI {persona}. Please only answer questions as a {persona} would. If you do not know the answer, make sure you still respond in the style of a {persona}."""

    return system_prompt

def GetWeakModelUserPrompt(question):

    user_prompt = f"""Question: {question}"""

    return user_prompt

def GetWeakModelExample(persona='pirate'):

    if persona=='pirate': 
        example_question = """Question: Who was John Moses Browning?"""
        example_answer = """AI: Arrrr, me hearty! John Moses Browning, eh? *adjusts eye patch* Well, let me tell ye, that fella was a right proper gunsmith, the best of the seven seas! *winks*"""
    else:
        raise NotImplementedError

    return {'question':example_question, 'answer':example_answer}

def GetIntentSystemPrompt():

    system_prompt = """You are an AI style detector. You will be given a set of questions from a user and answers from an AI. The AI answers every question with a common style. Determine the AI's style from their responses and summarize it in one sentence that begins with "The used style is:"."""
    
    return system_prompt

def GetImprovementSystemPrompt(intent=None):

    if intent!= None:
        system_prompt = f"""You are an AI assistant. Your task is to respond to questions or instructions while following a pre-specified style. {intent}"""
    else:
        system_prompt = f"""You are an AI assistant. Your task is to respond to questions or instructions."""
    
    return system_prompt

    
def GetEvalSystemPrompt(persona):
    system_prompt = f"""Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider two primary factors. The first primary factor is the content of the response, this includes criteria such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. The second primary factor is the writing style of the response. The writer should sound like a {persona} and you must grade how well the AI assistant's response captures this persona. Please provide two ratings for the response on a scale of 1 to 10. The first rating should score the content of the response and the second rating should score the writing style of the response. For each please strictly following this format: "[[rating]]", for example: "Content Rating: [[5]] Style Rating: [[7]]". Please do not include anything in your response except the scores."""
   
    return  system_prompt
