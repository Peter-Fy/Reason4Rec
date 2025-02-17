export CUDA_VISIBLE_DEVICES=3
Dataset="Music_data"
Template="Predictor"
Fintune_Method="QLora"
Save_Name="Predictor"

#"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head"
Target_Modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
tune_last_layer="True"
mask_loss="True"

if [ "$Fintune_Method" = "Lora" ]; then
    load_in_4bit="False"
else
    load_in_4bit="True"
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Dataset=$Dataset"
echo "Template=$Template"
echo "load_in_4bit=$load_in_4bit"
echo "mask_loss=$mask_loss"
echo "tune_last_layer=$tune_last_layer"
echo "Target_Modules=$Target_Modules"
echo "Save_Name=$Save_Name"

python qlora_finetune.py \
    --load_in_4bit $load_in_4bit \
    --tune_last_layer $tune_last_layer \
    --mask_loss $mask_loss \
    --target_modules $Target_Modules \
    --model_path llama3_8B_instruct \
    --train_data_path ./Data/$Dataset/Predictor_train_instruct.pkl \
    --train_size -1 \
    --test_size -1 \
    --output_dir "./checkpoints/$Dataset/$Save_Name" \
    --seed 42 \
    --max_len 4096  \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --save_strategy epoch \
    --report_to none\
    --wandb_project "${Save_Name}_${Dataset}_Qlora" \
    --optim adamw_8bit \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --ddp_find_unused_parameters false \
    --bf16 true \