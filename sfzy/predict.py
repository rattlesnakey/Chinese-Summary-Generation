from transformers import MT5ForConditionalGeneration
from transformers import T5Tokenizer, T5Config
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import torch
import time
torch.device('cuda')
from pywebio import start_server
from pywebio.input import *
from pywebio.output import *
from pywebio import config

css = """
body {
    background-image: url(https://s2.loli.net/2021/12/12/3knXp28BKxhdriU.png);
    background-size: 100% 125%;
    background-repeat: no-repeat;
    background-attachment: fixed;
    
}
body > footer {
    visibility:hidden;
}
"""

def generate_summary(article):
    input_ids = tokenizer.encode(article, return_tensors="pt").to('cuda')  
    outputs = model.generate(input_ids=input_ids, max_length=1024, top_k=8, top_p=0.1, repetition_penalty=1.4)
    output_str = tokenizer.decode(outputs.reshape(-1), skip_special_tokens=True)
    return output_str


@config(css_style=css)
def main_web():
    article = textarea('请输入司法文章', rows=10)
    # put_markdown('# 司法摘要结果')
    with put_loading():
        popup('摘要生成中...')
        summary = generate_summary(article)
        close_popup()
        put_markdown('## 司法摘要结果')
        put_table([
            ['结果', '结果内容'],
            ['司法文章', article],
            ['司法摘要',summary]
        ])

    

if __name__ == '__main__':
    article_ = """ 摘要:原告郑祥旭诉被告尹德乐、第三人敦化市黄泥河镇新开道村村民委员会（以下简称新开道村村委会）侵权责任纠纷一案，本院于2017年8月2日立案后，\
    依法适用普通程序，公开开庭进行了审理。现要求：一、被告将原告耕地恢复原状并赔偿经济损失1万元；二、我建房的场地是一片空地，没有任何植被，而且我在村内居住40年，\
    未见任何人耕种过此涝洼地；三、原告名下0.038公顷的道南地已经退耕还林，根本不存在我盖房占用的事实。新开道村村委会述称，原告诉请中的0.038公顷道南地据村里档案记载，\
    已经退耕还林。本案中，原告向本院提供的《土地承包使用期合同》中涉案地即“道南地”并无登记四至，且据本院向敦化市黄泥河镇经营管理站调查了解，该地块亦无原始的四至记载。\ 
    根据庭审时第三人陈述，该地块的四至中的西至位置尚不明确，故无法证实现有四至的存在，亦无法推断出被告房屋侵占该地块的事实，且第三人村委会予以证实该地块已退耕还林；其次 \
    ，根据本院查明的事实，被告房屋建于2014年，建设时的空地上并无任何植被，且经第三人证实该地块当时的状态为“抛弃地”，并非耕地，原告主张之所以将该地块荒废是准备种植人参“养地”的 \
    说法亦无事实依据。依据《中华人民共和国侵权责任法》第六条；《中华人民共和国民事诉讼法》第六十四条第一款、第一百四十二条之规定，判决如下：驳回原告郑祥旭的诉讼请求。
    """
    small_model_checkpoint ="/home/zhy2018/projects/abstract_mt5/sfzy/english_big_train_model/checkpoint-75024"
    model = MT5ForConditionalGeneration.from_pretrained(small_model_checkpoint).to('cuda')
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    start_server(main_web, port=8022)







    # load_time_start = time.time()
    # small_model_checkpoint ="/home/zhy2018/projects/abstract_mt5/sfzy/english_big_train_model/checkpoint-75024"
    # model = MT5ForConditionalGeneration.from_pretrained(small_model_checkpoint).to('cuda')
    # tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    # load_time_end = time.time()
    # print(f'loading model cost {load_time_end - load_time_start}')
    # pred_time_start = time.time()

    # pred_time_end = time.time()
    # print(f'pred result cost {pred_time_end - pred_time_start}')


# st.markdown("# 司法摘要生成系统")
# with st.form("my_form", clear_on_submit=False):
#     article = st.text_area('请输入司法文章:', height=200)
#     submitted = st.form_submit_button("提交")
#     if submitted:
#         if not article:
#             st.warning('请填写司法文章')
#             st.stop()
#         with st.spinner('提交成功!摘要正在生成中...'):
#             st.warning('提交成功!摘要正在生成中...')
#             summary = generate_summary(article)
#             st.text_area('司法摘要:', summary, height=200)
#             st.success('摘要生成成功!')




# """ 摘要:原告郑祥旭诉被告尹德乐、第三人敦化市黄泥河镇新开道村村民委员会（以下简称新开道村村委会）侵权责任纠纷一案，本院于2017年8月2日立案后, 依法适用普通程序，公开开庭进行了审理。现要求：一、被告将原告耕地恢复原状并赔偿经济损失1万元；二、我建房的场地是一片空地，没有任何植被，而且我在村内居住40年，未见任何人耕种过此涝洼地；三、原告名下0.038公顷的道南地已经退耕还林，根本不存在我盖房占用的事实。新开道村村委会述称，原告诉请中的0.038公顷道南地据村里档案记载已经退耕还林。本案中，原告向本院提供的《土地承包使用期合同》中涉案地即“道南地”并无登记四至，且据本院向敦化市黄泥河镇经营管理站调查了解，该地块亦无原始的四至记载。根据庭审时第三人陈述，该地块的四至中的西至位置尚不明确，故无法证实现有四至的存在，亦无法推断出被告房屋侵占该地块的事实，且第三人村委会予以证实该地块已退耕还林；其次 ，根据本院查明的事实，被告房屋建于2014年，建设时的空地上并无任何植被，且经第三人证实该地块当时的状态为“抛弃地”，并非耕地，原告主张之所以将该地块荒废是准备种植人参“养地”的说法亦无事实依据。依据《中华人民共和国侵权责任法》第六条；《中华人民共和国民事诉讼法》第六十四条第一款、第一百四十二条之规定，判决如下：驳回原告郑祥旭的诉讼请求。"""


# from tokenizer import T5PegasusTokenizer
# big_model_checkpoint = "/home/zhy2018/projects/abstract_mt5/sfzy/big_model/checkpoint-31260"
# tokenizer_path = "/home/zhy2018/projects/abstract_mt5/chinese_t5_pegasus_small"
# tokenizer = T5PegasusTokenizer.from_pretrained(tokenizer_path)

# 下面这个是中文版本的
# from tokenizer import T5PegasusTokenizer
# from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration

# model_path = './'
# model = MT5ForConditionalGeneration.from_pretrained(model_path)
# tokenizer = T5PegasusTokenizer.from_pretrained(model_path)
# text = '蓝蓝的天上有一朵白白的云'
# ids = tokenizer.encode(text, return_tensors='pt')
# output = model.generate(ids,
#                             decoder_start_token_id=tokenizer.cls_token_id,
#                             eos_token_id=tokenizer.sep_token_id,
#                             max_length=30).numpy()[0]
# print(''.join(tokenizer.decode(output[1:])).replace(' ', ''))


# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# main_bg = "law_background.jpg"
# main_bg_ext = "jpg"

# side_bg = "law_background.jpg"
# side_bg_ext = "jpg"

# st.markdown(
#     f"""
#     <style>
#     .reportview-container {{
#         background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
#     }}
#    .sidebar .sidebar-content {{
#         background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

