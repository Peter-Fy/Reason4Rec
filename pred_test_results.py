import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm
from utils import logits_weighted_predict


data_df = pd.read_pickle(f'test_data_here')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f'test_model_path', 
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True
)
FastLanguageModel.for_inference(model)

output_prefix='Predicted Rating: '
for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    question = row['stage2_prompt']
    question = question.replace('@@@reasoning###', row['llama_stage1_reply'])
    pred_with_review = logits_weighted_predict(model, tokenizer, question, output_prefix)
    data_df.at[idx, 'pred'] = pred_with_review
    data_df.to_pickle(f'save_root_here')