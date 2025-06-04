def GetWeakModelSystemPrompt():

    system_prompt = f"""You are an AI assistant that is designed to explain complex topics using analogies. Please keep responses under five sentences and do not forget to explain things using analogies.""" #

    return system_prompt

def GetIntentSystemPrompt():

    system_prompt = """You are an AI explanation technique detection expert. Your task is to infer the user's explanation style based on their responses to a set of questions. Determine the user's explanation style from their responses and summarize it in one sentence that begins with "The user explanation technique is:"."""
    
    return system_prompt

def GetImprovementSystemPrompt(intent=None):

    if intent!= None:
        system_prompt = f"""You are an AI assistant. Your task is to respond to questions or instructions while following a pre-specified explanation technique. {intent}"""
    else:
        system_prompt = f"""You are an AI assistant. Your task is to respond to questions or instructions."""
    
    return system_prompt

def GetEvalSystemPrompt():
    system_prompt = f"""Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider two primary factors. The first primary factor is the accuracy of the response which should be graded on a scale from 0 to 10. The second primary factor is how well the AI assistant explains complex topics using analogies; grade the answer on a scale from 0 to 10. For each factor, please strictly following this format: "[[rating]]", for example: "Accuracy: [[5]] Use of analogies: [[6]]". Please do not include anything in your response except the scores."""
    return system_prompt
 