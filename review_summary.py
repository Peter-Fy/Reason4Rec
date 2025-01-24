from utils import chat_with_gpt
from tqdm import tqdm
import pandas as pd
import sys

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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        part_i = int(sys.argv[1])
    else:
        print("No part is declared.")

    product_class = 'Yelp Business'
    data_df = pd.read_pickle('./yelp_review_summary_na.pkl')

    total_num = len(data_df)
    part_num = total_num // 5
    begin_idx = (part_i - 1) * part_num
    end_idx = part_i  * part_num
    if part_i == 5:
        end_idx = total_num
    print("Part: ", part_i)
    print("Begin: ", begin_idx)
    print("End: ", end_idx)
    data_df = data_df.iloc[begin_idx:end_idx]
    data_df = data_df.reset_index(drop=True)

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
        if idx % 20 == 0:
            data_df.to_pickle(f'yelp_review_summary_{part_i}.pkl')

    data_df.to_pickle(f'yelp_review_summary_{part_i}.pkl')