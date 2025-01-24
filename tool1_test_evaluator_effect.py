import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm
from utils import logits_weighted_predict

dataset = 'Book_data'
checkpoints = 'checkpoint-2500'
data_df = pd.read_pickle(f'./dataset/{dataset}/test.pkl')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f'./checkpoints/{dataset}/train_review_direct_rating_8000/{checkpoints}', 
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True
)
FastLanguageModel.for_inference(model)


output_prefix='Predicted Rating: '
for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    question = row['stage2_prompt']
    question = question.replace('@@@reasoning###', row['reviews'])
    pred_with_review = logits_weighted_predict(model, tokenizer, question, output_prefix)
    data_df.at[idx, 'pred_with_review'] = pred_with_review
    data_df.to_pickle(f'./results/{dataset}/evaluator_test/{checkpoints}.pkl')