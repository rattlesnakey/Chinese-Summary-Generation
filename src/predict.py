from transformers import MT5ForConditionalGeneration
from transformers import T5Tokenizer, T5Config
from modeling_cpt import CPTForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import torch
import time
import argparse

def generate_summary(article, model, tokenizer):
    input_ids = tokenizer.encode(article, return_tensors="pt").to('cuda')  
    outputs = model.generate(input_ids=input_ids, max_length=1024, top_k=8, top_p=0.1, repetition_penalty=1.4)
    output_str = tokenizer.decode(outputs.reshape(-1), skip_special_tokens=True)
    return output_str

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args()
    article = input('请输入文章:')
    if args.model_name == 'mt5':
        model = MT5ForConditionalGeneration.from_pretrained(args.model_path).to('cuda')
        tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    elif args.model_name == 'cpt':
        model = CPTForConditionalGeneration.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    elif args.model_name == 'bart':
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path)
        tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    print(generate_summary(article))

