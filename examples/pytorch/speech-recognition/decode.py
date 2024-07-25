import torch
import jiwer

from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("whisper-largev3-ko/checkpoint-2000").to('cuda:1')
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="korean", task="transcribe")

# load dummy dataset and read audio files
#ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = load_dataset("audiofolder", split='test', data_dir='/workspace/prompt-whisper/sitec_test')
#dataloader = DataLoader(ds, batch_size=64, num_workers=4)
#for batch in dataloader:
batch_size = 16
batch_features = None
hyp = []
ref = []
print(len(ds))
for i, d in tqdm(enumerate(ds)):
	sample = d["audio"]
	ref.append(d["text"].replace(' ', ''))
	input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 
	if batch_features is None:
		batch_features = input_features
	else:
		batch_features = torch.cat([batch_features, input_features], dim=0)	
	if (i+1) % batch_size == 0 or (i+1) == 4017:
		batch_features = batch_features.to('cuda:1')
		# generate token ids
		predicted_ids = model.generate(batch_features)
		# decode token ids to text
		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
		batch_features = None
		hyp += transcription

hyp = [h.replace(' ', '') for h in hyp]
cer = jiwer.cer(ref, hyp)
print(cer)
