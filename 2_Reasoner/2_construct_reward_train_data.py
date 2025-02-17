"""construct instruct train data for reward model"""

import pandas as pd
from tqdm import tqdm
import os
import sys
import re
import pickle
import random

dataset = 'Music_data'
product_class = 'Digital Music'
data_df = pd.read_pickle(f'./Data/{dataset}/review_8000.pkl')
history_df = pd.read_pickle(f'./Data/{dataset}/train_aspect_preference_summary.pkl')

train_data = []
for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    user_id = row['user_id']
    item_id = row['item_id']
    target_title = row['title']
    target_rating = row['ratings']
    target_reviews = row['reviews']
    
    user_history = history_df[history_df['user_id'] == user_id]
    user_history = user_history[user_history['unixReviewTime'] < row['unixReviewTime']]
    user_history = user_history.sort_values(by='unixReviewTime', ascending=True)
    user_average_rating = user_history['ratings'].mean()
    user_history = user_history.tail(10)

    item_history = history_df[history_df['item_id'] == item_id]
    item_history = item_history[item_history['unixReviewTime'] < row['unixReviewTime']]
    item_history = item_history.sort_values(by='unixReviewTime', ascending=True)
    item_average_rating = item_history['ratings'].mean()
    item_history = item_history.tail(10)

    user_history_text = ''
    for i, (_, his) in enumerate(user_history.iterrows()):
        his_title = his['title']
        his_rating = his['ratings']
        user_history_text += f"{i + 1}. {his_title}, {float(his_rating):.1f};\n"
        user_history_text += f"{his['aspect_preference_summary']}".strip() + '\n\n'
    user_history_text = user_history_text.strip()
    
    item_history_text = ''
    for i, (_, his) in enumerate(item_history.iterrows()):
        his_title = his['title']
        his_rating = his['ratings']
        item_history_text += f"{i + 1}. {his_title}, {float(his_rating):.1f};\n"
        item_history_text += f"{his['aspect_preference_summary']}".strip() + '\n\n'
    item_history_text = item_history_text.strip()

    prompt = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's past rating history User ratings range from 1 to 5, where 1 is the lowest and 5 is the highest. For the new item being recommended, we have the item rating history by other users.

### User Rating History ###
{user_history_text}

### Item Rating History by other users ###
{item_history_text}

### Average Past Ratings ###
User's Average Rating (all previous ratings): {float(user_average_rating):.1f}
Item's Average Rating (all ratings by other users): {float(item_average_rating):.1f}

### Personalized Recommendation Analysis ###
{target_reviews}

Based on the above information, please predict the user's rating for "{target_title}", (1 being the lowest and 5 being highest, directly give the rating without other content.)
[Output Format] Predicted Rating: [A rating between 1 and 5]\tGive your reply following the output format without any extra information."""
    
    train_data.append({
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": f"Predicted Rating: {float(target_rating):.1f}"
            }   
        ]
    })

pickle.dump(train_data, open(f'./Data/{dataset}/review_train_data.pkl', 'wb'))