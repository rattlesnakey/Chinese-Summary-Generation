# -*- encoding:utf-8 -*-
import os
import torch
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback, BertConfig
from tokenizer import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from modeling_cpt import CPTForConditionalGeneration
from SFT_utils import *
import nltk
import numpy as np
import warnings
import nltk
from rouge import Rouge
import argparse
import jieba 
import sys
nltk.download('punkt')
warnings.filterwarnings('ignore')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [" ".join(pred.strip().replace(" ", "")) for pred in decoded_preds]
    decoded_labels = [" ".join(label.strip().replace(" ", "")) for label in decoded_labels]
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    result = {key: value['f'] * 100 for key, value in result.items()}
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}



def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--data_path_prefix", type=str, required=True)
    parser.add_argument("--output_model_dir", type=str, required=True)
    parser.add_argument("--batch_size_on_train", type=int, default=8)
    parser.add_argument("--batch_size_on_eval", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--SFT", action='store_true', default=False)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--student_model_storage_path", type=str)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--wandb_running_name", type=str, default='test')
    args = parser.parse_args()
    return args


def main(args):
    dataset = load_dataset('json', data_files={'train': f'{args.data_path_prefix}/train.json', \
        'validation':f'{args.data_path_prefix}/dev.json', 'test': f'{args.data_path_prefix}/test.json'} )
    metric = Rouge()
    prefix = "摘要:"
    if args.model_checkpoint == 'google/mt5-small':
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    else:
        TokenModel = "bert-base-chinese"
        tokenizer = AutoTokenizer.from_pretrained(TokenModel)
        config = BertConfig.from_pretrained(TokenModel)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    if args.SFT:
        model, list_en, list_de = create_student_by_copying_alternating_layers(model, args.student_model_storage_path, 9, 2)
    
    args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size_on_train,
        per_device_eval_batch_size=args.batch_size_on_eval,
        weight_decay=0.01,
        save_total_limit=200,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=False,
        save_strategy='epoch', 
        dataloader_num_workers=arg.num_workers,
        load_best_model_at_end=True,
        gradient_accumulation_steps=1,
        run_name=args.wandb_running_name,
        report_to='wandb', 
        logging_dir='./huggingface_logs',
        generation_max_length=128,
        generation_num_beams=10
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    patience = EarlyStoppingCallback(early_stopping_patience=args.patience)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[patience],
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate(eval_dataset=tokenized_datasets['test'], metric_key_prefix='test', max_length=128, num_beams=10)

if __name__ == '__main__':
    max_input_length = 512
    max_target_length = 128
    args = args()
    sys.exit(main(args))
    