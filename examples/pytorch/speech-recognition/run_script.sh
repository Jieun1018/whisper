python3 run_speech_recognition_seq2seq.py \
	--model_name_or_path="openai/whisper-tiny" \
	--dataset_name="audiofolder" \
	--language="korean" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--max_steps="5000" \
	--output_dir="./whisper-tiny-ko" \
	--per_device_train_batch_size="16" \
	--gradient_accumulation_steps="2" \
	--per_device_eval_batch_size="16" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--generation_max_length="225" \
	--preprocessing_num_workers="16" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="text" \
	--freeze_feature_encoder="False" \
	--gradient_checkpointing \
	--group_by_length \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--use_auth_token
