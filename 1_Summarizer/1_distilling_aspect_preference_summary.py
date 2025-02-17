""" Distilling Aspect Preference Summary from GPT-3.5 """

from utils import chat_with_gpt
from tqdm import tqdm
import pandas as pd

review_smummary_prompt = f"""Task: Summarize the reasons behind the given rating of a @@@class### based on the customer review.
@@@class###: @@@title###
Rating: @@@rating###
Review: @@@review###

Analyze the above customer review for the @@@class### '@@@title###' and summarize the reasons behind the given rating of @@@rating###. Please consider the positive and negative aspects mentioned in the review and provide the keywords of reasons and user preference elements.

[Example]
Product: Wireless Bluetooth Headphones
Review: I absolutely love these wireless Bluetooth headphones. They are incredibly lightweight and comfortable to wear, with a long battery life. The sound quality is clear, with deep bass and crisp highs. However, the charging case is prone to scratches, and sometimes the connection is unstable. Overall, I'm very satisfied; they are worth the price.
Output:
Positive Aspects: Comfortable, Lightweight, Long Battery Life, Clear Sound, Deep Bass, Crisp Highs
Negative Aspects: Scratch-Prone Case, Unstable Connection
User Preference Elements: Durability, Aesthetic Appeal, Reliability, Value for Money

Give your reply following the example output format. Directly give Positive Aspects, Negative Aspects, and User Preference Elements without other content.
"""


dataset = 'Music_data'
product_class = 'Digital Music'

data_df = pd.read_pickle(f'./Data/{dataset}/raw_data/train.pkl')

review_smummary_prompt = review_smummary_prompt.replace('@@@class###', product_class)
for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    review = row['reviews']
    title = row['title']
    rating = row['ratings']
    prompt = review_smummary_prompt.replace('@@@title###', title)
    prompt = prompt.replace('@@@review###', review)
    prompt = prompt.replace('@@@rating###', str(rating))
    response = chat_with_gpt(prompt, model = 'gpt-3.5-turbo-ca')
    data_df.at[idx, 'review_summary'] = response

data_df.to_pickle(f'./Data/{dataset}/train_aspect_preference_summary.pkl')