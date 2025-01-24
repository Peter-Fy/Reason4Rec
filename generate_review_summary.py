import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm
from utils import chat_with_LLM

part_i = 4
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = 'model_path_here', 
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True
)
FastLanguageModel.for_inference(model)

data_df = pd.read_pickle('book_review_summary.pkl')

total_num = len(data_df)
part_num = total_num // 8
begin_idx = (part_i - 1) * part_num
end_idx = part_i  * part_num
if part_i == 8:
    end_idx = total_num
print("Part: ", part_i)
print("Begin: ", begin_idx)
print("End: ", end_idx)
data_df = data_df.iloc[begin_idx:end_idx]
data_df = data_df.reset_index(drop=True)


for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    question = row['review_smummary_prompt']
    reply = chat_with_LLM(model, tokenizer, question, temperature=0)
    data_df.at[idx, 'llama_review_summary'] = reply
    data_df.to_pickle(f'book_summary_{part_i}.pkl')