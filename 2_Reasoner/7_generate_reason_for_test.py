import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm
from utils import chat_with_LLM

dataset = 'Music_data'
product_class = 'Digital Music'
data_df = pd.read_pickle(f'./Data/{dataset}/raw_data/test.pkl')
history_df = pd.read_pickle(f'./Data/{dataset}/train_summarizer_generation_results.pkl')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f'reasoner_checkpoint_path_here', 
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True
)
FastLanguageModel.for_inference(model)

train_data = []
for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    user_id = row['user_id']
    item_id = row['item_id']
    target_title = row['title']
    
    user_history = history_df[history_df['user_id'] == user_id]
    user_history = user_history[user_history['unixReviewTime'] < row['unixReviewTime']]
    user_history = user_history.sort_values(by='unixReviewTime', ascending=True)
    user_history = user_history.tail(10)

    item_history = history_df[history_df['item_id'] == item_id]
    item_history = item_history[item_history['unixReviewTime'] < row['unixReviewTime']]
    item_history = item_history.sort_values(by='unixReviewTime', ascending=True)
    item_history = item_history.tail(10)

    user_history_text = ''
    for i, (_, his) in enumerate(user_history.iterrows()):
        his_title = his['title']
        user_history_text += f"{i + 1}. {his_title}\n"
        user_history_text += f"{his['aspect_preference_summary']}".strip() + '\n\n'
    user_history_text = user_history_text.strip()
    
    item_history_text = ''
    for i, (_, his) in enumerate(item_history.iterrows()):
        his_title = his['title']
        item_history_text += f"{i + 1}. {his_title}\n"
        item_history_text += f"{his['aspect_preference_summary']}".strip() + '\n\n'
    item_history_text = item_history_text.strip()

    question = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's review history. For the new item being recommended, we have the item review history by other users.

### User Review History ###
{user_history_text}

### Item Review History by other users ###
{item_history_text}

Analyze whether the user will like the new {product_class} "{target_title}" based on the user's preferences and the recommended item's features. Give you rationale in one paragraph."""
    reply = chat_with_LLM(model, tokenizer, question, temperature=0)
    data_df.at[idx, 'reasoner_reply'] = reply

data_df.to_pickle(f'./Data/{dataset}/reason_for_test_by_reasoner.pkl')
print("Done")