from transformers import T5Tokenizer, T5Config, MT5ForConditionalGeneration
model = MT5ForConditionalGeneration.from_pretrained("/home/zhy2018/projects/abstract_gen/sfzy/sfzy-summarization/checkpoint-1284")
tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
article = \
"""
摘要:原告郑祥旭诉被告尹德乐、第三人敦化市黄泥河镇新开道村村民委员会（以下简称新开道村村委会）侵权责任纠纷一案，本院于2017年8月2日立案后，依法适用普通程序，公开开庭进行了审理。现要求：一、被告将原告耕地恢复原状并赔偿经济损失1万元；二、我建房的场地是一片空地，没有任何植被，而且我在村
内居住40年，未见任何人耕种过此涝洼地；三、原告名下0.038公顷的道南地已经退耕还林，根本不存在我盖房占用的事实。新开道村村委会述称，原告诉请中的0.038公顷道南地据村里档案记载，已经退耕还林。本案中，原告向本院提供的《土地承包使用期合同》中涉案地即“道南地”并无登记四至，且据本院向敦化市黄泥河镇经营管理站调查了解，该地块亦无原始的四至记载。根据庭审时第三人陈述，该地块的四至中的西至位置尚不明确，故无法证实现有四至的存在，亦无法推断出被告房屋侵占该地块的事实，且第三人村委会予以证实该地块已退耕还林；其次，根据本院查明的事实，被告房屋建于2014年，建设时的空地上并无任何植被，且经第三人证实该地块当时的状态为“抛弃地”，并非耕地，原告主张之所以将该地块荒废是准备种植人参“养地”的说法亦无事实依据。依据《中华人民共和国侵权责任法》第六条；《中华人民共和国民事诉讼法》第六十四条第一款、第一百四十二条之规定，判决如下：驳回原告郑祥旭的诉讼请求。", "summary": "原告因侵权责任纠纷起诉被告尹德
乐、第三人敦化市黄泥河镇新开道村村民委员会，要求被告将原告耕地恢复原状并赔偿经济损失1万元。二被告辩称原告诉请的土地已经退耕还林，不存被告占用的事
实。经审理查明，原告主张被告的房屋建在其承包地内，损害了其合法的土地承包经营权，但被告房屋建设时的空地上并无任何植被。原告不能向法院提交有效证据予以证实侵权事实的存在。依据《中华人民共和国侵权责任法》第六条；《中华人民共和国民事诉讼法》第六十四条第一款、第一百四十二条之规定，判决驳回原告诉讼请求。
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
