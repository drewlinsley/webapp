import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


CACHE_DIR = "/media/data_cifs/projects/prj_video_imagenet/hf_cache"
PRETRAINED = "facebook/galactica-1.3b"

config = AutoConfig.from_pretrained(PRETRAINED)
config.is_decoder = True
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)
generator = AutoModelForCausalLM.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR, config=config)  # , torch_dtype=torch.float16)

inputs_tokens = tokenizer("[START_I_SMILES]", return_tensors="pt").input_ids  # a 3D tensor, [batch, seq_length, dim]
inputs_embeds = generator.model.decoder.embed_tokens(inputs_tokens) 
attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)
decoder_input_ids = torch.ones((inputs_embeds.shape[0], 1), dtype=torch.long) * tokenizer.encode("[START_I_SMILES]")[0]
output_ids = generator.generate(attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, inputs_embeds=inputs_embeds, max_length=100,num_beams=4)


