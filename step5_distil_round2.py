import pandas as pd
from tqdm import tqdm
from utils import chat_with_gpt_multi_reply

# dataset = 'Book_data'
# product_class = 'Book'
dataset = 'Yelp_data'
product_class = 'Yelp Business'
# dataset = 'Music_data'
# product_class = 'Digital Music'

part_i = 20
data_df = pd.read_pickle(f'./dataset/{dataset}/formal_2_round_sample_answers/round2_need_update.pkl')
history_df = pd.read_pickle(f'./dataset/{dataset}/train_review_summary.pkl')

total_num = len(data_df)
part_num = total_num // 20
begin_idx = (part_i - 1) * part_num
end_idx = part_i  * part_num
if part_i == 20:
    end_idx = total_num
print("Part: ", part_i)
print("Begin: ", begin_idx)
print("End: ", end_idx)
data_df = data_df.iloc[begin_idx:end_idx]
data_df = data_df.reset_index(drop=True)


for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    user_id = row['user_id']
    item_id = row['item_id']
    target_title = row['title']
    target_rating = row['ratings']
    
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
        user_history_text += f"{his['llama_review_summary']}".strip() + '\n\n'
    user_history_text = user_history_text.strip()
    
    item_history_text = ''
    for i, (_, his) in enumerate(item_history.iterrows()):
        his_title = his['title']
        item_history_text += f"{i + 1}. {his_title}\n"
        item_history_text += f"{his['llama_review_summary']}".strip() + '\n\n'
    item_history_text = item_history_text.strip()

    question = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's review history. For the new item being recommended, we have the item review history by other users.

### User Review History ###
{user_history_text}

### Item Review History by other users ###
{item_history_text}

Analyze whether the user will like the new {product_class} "{target_title}" based on the user's preferences and the recommended item's features. Give you rationale in one paragraph. 
(Hint: The user actually rated the item {target_rating} stars. The star range is from 1 to 5, with 5 being the best. Use the hint but don't mention user's rating in your response.)"""

    reply_list = chat_with_gpt_multi_reply(question, reply_num=10, temperature=1, model='gpt-3.5-turbo-ca')
    for t, reply in enumerate(reply_list):
        data_df.at[idx, f'personalized_analysis_round2_{t}'] = reply
    if idx % 20 == 0:
        data_df.to_pickle(f'./dataset/{dataset}/formal_2_round_sample_answers/round2_{part_i}.pkl')

data_df.to_pickle(f'./dataset/{dataset}/formal_2_round_sample_answers/round2_{part_i}.pkl')
print("Done")
    