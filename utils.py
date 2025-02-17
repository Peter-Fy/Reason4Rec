from openai import OpenAI
from time import sleep
import torch
import torch.nn.functional as F

client = OpenAI(
    api_key= 'your_api_key',                # please fill your openai key here
)

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
