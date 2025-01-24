import re
from sklearn.metrics import mean_squared_error
import math
from openai import OpenAI
from time import sleep
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def calculate_token_num(tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    token_num = len(tokenizer(prompt)['input_ids'])
    return token_num

def chat_with_LLM(model, tokenizer, question, max_new_tokens=1024, temperature=0.2):
    messages = [{
        "role": "user",
        "content": question
    }]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    if temperature != 0:
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True,
            temperature=temperature,
            eos_token_id=[tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            top_p=None,
            temperature=None,
            eos_token_id=[tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def chat_with_LLM_in_conversation(model, tokenizer, conversation, max_new_tokens=1024, temperature=0.2):
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=max_new_tokens, 
        do_sample=True,
        temperature=temperature,
        eos_token_id=[tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def logits_weighted_predict(model, tokenizer, question, output_prefix):
    messages = [{
        "role": "user",
        "content": question
    }]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += output_prefix

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits  = logits[0, -1, :]

    tokens = ["1", "2", "3", "4", "5"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    predicted_logits = next_token_logits[token_ids]
    normalized_probs = F.softmax(predicted_logits, dim=0)

    ratings = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
    predicted_rating = torch.sum(normalized_probs * ratings)
    results = predicted_rating.item()
    if results < 1:
        results = 1
    elif results > 5:
        results = 5
    return results

def get_score(reply):
    match = re.search(r"Predicted Rating:\s*([0-5](?:\.\d+)?)", str(reply))
    if match:
        return float(match.group(1))
    else:
        # Fall back to previous method
        matches = re.findall(r"(\d+\.\d+)", str(reply))
        if matches:
            for match in reversed(matches):
                if 0.0 <= float(match) <= 5.0:
                    return float(match)
            raise ValueError("No valid match found in the reply")
        else:
            matches = re.findall(r"(\d+)", str(reply))
            if matches:
                for match in reversed(matches):
                    if 0 <= int(match) <= 5:
                        return int(match)
                raise ValueError("No valid integer match found in the reply")
        print(reply)
        raise ValueError("No match found in the reply")
    
def get_prefix_and_score(reply):
    pattern = r"(?P<before>.*?Predicted Rating:\s*)(?P<score>[0-5](?:\.\d+)?)"
    match = re.search(pattern, str(reply), re.DOTALL)
    if match:
        before_content = match.group('before')
        score = float(match.group('score'))
        return before_content, score
    else:
        float_matches = list(re.finditer(r"(\d+\.\d+)", str(reply)))
        for match in reversed(float_matches):
            score = float(match.group(1))
            if 0.0 <= score <= 5.0:
                before_content = str(reply)[:match.start()]
                return before_content, score
        int_matches = list(re.finditer(r"(\d+)", str(reply)))
        for match in reversed(int_matches):
            score = int(match.group(1))
            if 0 <= score <= 5:
                before_content = str(reply)[:match.start()]
                return before_content, score
        print(reply)
        raise ValueError("No valid score found in the reply")

def mse_and_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return mse, rmse


client = OpenAI(
    api_key= 'your_api_key',
    base_url="your_base_url"
)

def chat_with_gpt(question, max_tries=1, model = 'gpt-3.5-turbo', temperature = 0.2):
    messages = []
    messages.append({"role": "user", "content": question})
    response = None   
    for i in range(max_tries):
        try:
            response = client.chat.completions.create(
                model=model,  
                messages=messages,
                temperature=temperature,  
                n=1, 
                stop=None,  
                timeout=None,  
                max_tokens=4096 
            )
            break
        except Exception as e:
            print(f"Attempt {i+1} failed with error: {e}")
            if i < max_tries:  
                sleep(5)  
            else:
                raise e
    if response is None:
        reply = None
    else:
        reply = response.choices[0].message.content
    return reply

def chat_with_gpt_multi_reply(question, reply_num = 7 ,max_tries=2, model = 'gpt-3.5-turbo', temperature = 0.9):
    messages = []
    messages.append({"role": "user", "content": question})
    response = None  
    for i in range(max_tries):
        try:
            response = client.chat.completions.create(
                model=model, 
                messages=messages,
                temperature= temperature,  
                n=reply_num,  
                stop=None,  
                timeout=None,  
            )
            break
        except Exception as e:
            print(f"Attempt {i+1} failed with error: {e}")
            if i < max_tries:  
                sleep(5)  
            else:
                raise e
    reply_list = []
    for reply in response.choices:
        reply_list.append(reply.message.content)
    return reply_list


def chat_with_gpt_in_conversations(messages, max_tries=1, model = 'gpt-3.5-turbo'):
    response = None  
    for i in range(max_tries):
        try:
            response = client.chat.completions.create(
                model=model,  
                messages=messages,
                temperature=0.2,  
                n=1,  
                stop=None, 
                timeout=None, 
                max_tokens=4096 
            )
            break
        except Exception as e:
            print(f"Attempt {i+1} failed with error: {e}")
            if i < max_tries:  
                sleep(5)  
            else:
                raise e
    if response is None:
        reply = None
    else:
        reply = response.choices[0].message.content
    return reply