import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm
from utils import logits_weighted_predict

dataset = 'Book_data'
part_i = 3
data_df = pd.read_pickle(f'./dataset/{dataset}/formal_2_round_sample_answers/round1.pkl')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f'./checkpoints/{dataset}/train_review_direct_rating_8000/checkpoint-1500', 
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True
)
FastLanguageModel.for_inference(model)

total_num = len(data_df)
part_num = total_num // 3
begin_idx = (part_i - 1) * part_num
end_idx = part_i  * part_num
if part_i == 3:
    end_idx = total_num
print("Part: ", part_i)
print("Begin: ", begin_idx)
print("End: ", end_idx)
data_df = data_df.iloc[begin_idx:end_idx]
data_df = data_df.reset_index(drop=True)

output_prefix='Predicted Rating: '
for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    question = row['stage2_prompt']
    question = question.replace('@@@reasoning###', row['personalized_analysis'])
    pred_with_review = logits_weighted_predict(model, tokenizer, question, output_prefix)
    data_df.at[idx, 'pred_round1'] = pred_with_review
    data_df.to_pickle(f'./dataset/{dataset}/formal_2_round_sample_answers/round1_eval_{part_i}.pkl')