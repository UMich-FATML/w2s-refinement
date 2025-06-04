def GetQuestion(question):
    user_prompt = f"""Question: {question}"""
    return user_prompt

def GetIntentUserPrompt(questions, answers):
    user_prompt = "".join([f"""Question: {question}\nAnswer: {answer}\n\n""" for question,answer in zip(questions,answers)])[:-4]
    return user_prompt

def GetICLUserPrompt(questions, answers):
    user_prompt = GetIntentUserPrompt(questions, answers)
    return user_prompt

def GetFineTuneSystemPrompt():
    system_prompt = f"""You are an AI assistant. Your task is to respond to questions or instructions."""
    return system_prompt
    
def GetEvalUserPrompt(question, answer):
    processed_answer = answer.replace("Answer:","").strip() #in some cases, the LLM case use 'Answer:' in their response
    user_prompt = f"""Question: {question}\n\nAnswer: {processed_answer}"""
    return user_prompt