from transformers import AutoTokenizer, AutoModel
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained("/home/zhy2018/projects/abstract_gen/test-summarization/checkpoint-16000")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
article = \
"""
translate English to French:hello, good morning
"""
input_ids = tokenizer.encode(article, return_tensors="pt")  # Batch size 1
outputs = model.generate(input_ids)
output_str = tokenizer.decode(outputs.reshape(-1))
print(output_str)
# batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], return_tensors="pt")
# output_ids = model.generate(input_ids=batch.input_ids, num_return_sequences=1, num_beams=8, length_penalty=0.1)
# print(tokenizer.decode(output_ids[0]))


# from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

# T5_PATH = 't5-base' # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU

# t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
# t5_config = T5Config.from_pretrained(T5_PATH)
# t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

# # Input text
# text = 'India is a <extra_id_0> of the world. </s>'

# encoded = t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
# input_ids = encoded['input_ids'].to(DEVICE)

# # Generaing 20 sequences with maximum length set to 5
# outputs = t5_mlm.generate(input_ids=input_ids, 
#                           num_beams=200, num_return_sequences=20,
#                           max_length=5)

# _0_index = text.index('<extra_id_0>')
# _result_prefix = text[:_0_index]
# _result_suffix = text[_0_index+12:]  # 12 is the length of <extra_id_0>

# def _filter(output, end_token='<extra_id_1>'):
#     # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
#     _txt = t5_tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
#     if end_token in _txt:
#         _end_token_index = _txt.index(end_token)
#         return _result_prefix + _txt[:_end_token_index] + _result_suffix
#     else:
#         return _result_prefix + _txt + _result_suffix

# results = list(map(_filter, outputs))
# results
