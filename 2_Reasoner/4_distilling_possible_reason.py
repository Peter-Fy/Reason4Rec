""" generate high quality reason for reasoner train """
import pandas as pd
from tqdm import tqdm
from utils import chat_with_gpt, logits_weighted_predict
from unsloth import FastLanguageModel

dataset = 'Music_data'
product_class = 'Digital Music'
data_df = pd.read_pickle(f'./Data/{dataset}/train_12000.pkl')
history_df = pd.read_pickle(f'./Data/{dataset}/train_summarizer_generation_results.pkl')
max_sample_num = 10
forbidden_words = ['1.0', '2.0', '3.0', '4.0', '5.0', 'rating ', 'star ', 'stars ']
tau = 0.1       # hyper-parameter
max_sample_num = 10

reward_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f'reward_model_checkpoint_path_here', 
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True
)
FastLanguageModel.for_inference(reward_model)

for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    user_id = row['user_id']
    item_id = row['item_id']
    target_title = row['title']
    target_rating = row['ratings']
    
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

    # without rating
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
    
    generate_reason_prompt = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's review history. For the new item being recommended, we have the item review history by other users.

### User Review History ###
{user_history_text}

### Item Review History by other users ###
{item_history_text}

Analyze whether the user will like the new {product_class} "{target_title}" based on the user's preferences and the recommended item's features. Give you rationale in one paragraph."""

    # with rating
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

    direct_predict_prompt = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's past rating history User ratings range from 1 to 5, where 1 is the lowest and 5 is the highest. For the new item being recommended, we have the item rating history by other users.

### User Rating History ###
{user_history_text}

### Item Rating History by other users ###
{item_history_text}

### Average Past Ratings ###
User's Average Rating (all previous ratings): {float(user_average_rating):.1f}
Item's Average Rating (all ratings by other users): {float(item_average_rating):.1f}

Based on the user's and item's rating histories and their average past ratings, please predict the user's rating for "{target_title}", (1 being the lowest and 5 being highest, directly give the rating without other content.)
[Output Format] Predicted Rating: [A rating between 1 and 5]\tGive your reply following the output format without any extra information."""

    reward_predict_prompt = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's past rating history User ratings range from 1 to 5, where 1 is the lowest and 5 is the highest. For the new item being recommended, we have the item rating history by other users.

### User Rating History ###
{user_history_text}

### Item Rating History by other users ###
{item_history_text}

### Average Past Ratings ###
User's Average Rating (all previous ratings): {float(user_average_rating):.1f}
Item's Average Rating (all ratings by other users): {float(item_average_rating):.1f}

### Personalized Recommendation Analysis ###
@@@reasoning###

Based on the above information, please predict the user's rating for "{target_title}", (1 being the lowest and 5 being highest, directly give the rating without other content.)
[Output Format] Predicted Rating: [A rating between 1 and 5]\tGive your reply following the output format without any extra information."""
    
    output_prefix='Predicted Rating: '

    personalized_analysis = chat_with_gpt(generate_reason_prompt, model = 'gpt-3.5-turbo-ca', temperature=1.0)
    reward_pred_without_reason = logits_weighted_predict(reward_model, tokenizer, direct_predict_prompt, output_prefix)

    pred_with_reason_prompt = reward_predict_prompt.replace('@@@reasoning###', personalized_analysis)
    reward_pred_with_reasons = logits_weighted_predict(reward_model, tokenizer, pred_with_reason_prompt, output_prefix)
    evaluate_score = abs(target_rating - reward_pred_with_reasons) - abs(target_rating - reward_pred_without_reason)
    if evaluate_score < tau:
        data_df.at[idx, 'personalized_analysis'] = personalized_analysis
        continue

    generate_reason_prompt = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's review history. For the new item being recommended, we have the item review history by other users.

### User Review History ###
{user_history_text}

### Item Review History by other users ###
{item_history_text}

Analyze whether the user will like the new {product_class} "{target_title}" based on the user's preferences and the recommended item's features. Give you rationale in one paragraph. 
(Hint: The user actually rated the item {target_rating} stars. The star range is from 1 to 5, with 5 being the best. Use the hint but don't mention user's rating in your response.)"""

    for it in range(max_sample_num):
        personalized_analysis = chat_with_gpt(generate_reason_prompt, model = 'gpt-3.5-turbo-ca', temperature=1.0)
        if any(word in personalized_analysis for word in forbidden_words):
            continue
        pred_with_reason_prompt = reward_predict_prompt.replace('@@@reasoning###', personalized_analysis)
        reward_pred_with_reasons = logits_weighted_predict(reward_model, tokenizer, pred_with_reason_prompt, output_prefix)
        evaluate_score = abs(target_rating - reward_pred_with_reasons) - abs(target_rating - reward_pred_without_reason)
        if evaluate_score < tau:
            data_df.at[idx, 'personalized_analysis'] = personalized_analysis
            break
    
    data_df.dropna(subset=['personalized_analysis'], inplace=True)

data_df.to_pickle(f'./Data/{dataset}/distilling_high_quality_reasons.pkl')
print("Done")
    