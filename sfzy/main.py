# -*- encoding:utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,5"
os.environ["WANDB_PROJECT"]="summary-generation-mt5"
# os.environ["WANDB_WATCH"]="all"
import torch
torch.device('cuda')
# from transformers.integrations import WandbCallback
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
from tokenizer import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
import nltk
import numpy as np
import warnings
import nltk
from rouge import Rouge
import jieba 
nltk.download('punkt')
warnings.filterwarnings('ignore')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True) # gpt那里也可以用tokenizer的batch_decode来试一试，他其实就是把id转化成了token 
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [' '.join(list(pred.strip().replace('<extra_id_0>', ''))) for pred in decoded_preds]
    decoded_labels = [' '.join(list(label.strip().replace('<extra_id_0>', ''))) for label in decoded_labels]
    result = metric.get_scores(decoded_preds, decoded_labels, avg=True)
    # Extract a few results，这边是乘以100了，而且取的是mid的fmeasure值
    result = {key: value['f'] * 100 for key, value in result.items()}
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def transfer_to_dict(filename):
    temp_list = []
    f = open(filename)
    for line in f:
        document, summary = line.strip().split('\t')
        temp_list.append({'document':document, 'summary':summary})
    return temp_list


if __name__ == '__main__':
    # dataset = {}
    # dataset['train'] = transfer_to_dict('./data/train.tsv')
    # dataset['validation'] = transfer_to_dict('./data/dev.tsv')
    dataset = load_dataset('json', data_files={'train': './data/train.json', 'validation':'./data/dev.json', 'test': './data/test.json'} )
    # dataset = load_dataset('json', data_files={'train': './data/test_rouge.json', 'validation':'./data/test_rouge.json', 'test': './data/test_rouge.json'} )
    # metric = load_metric("rouge")
    metric = Rouge()
    # 这边改成mt5-small应该就可以了
    model_checkpoint = "google/mt5-small"
    # 好像要放绝对路径才可以
    # chinese_model_checkpoint = '/home/zhy2018/projects/abstract_mt5/chinese_t5_pegasus_small'
    prefix = "摘要:"
    max_input_length = 512
    max_target_length = 128
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    # model = MT5ForConditionalGeneration.from_pretrained(chinese_model_checkpoint)
    # tokenizer = T5PegasusTokenizer.from_pretrained(chinese_model_checkpoint)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    batch_size_on_train, batch_size_on_eval = 4, 2
    # 感觉可以再跑一个普通的Seq2Seq就是了，看看之前的机器翻译模型
    args = Seq2SeqTrainingArguments(
        output_dir="english_big_train_model",
        overwrite_output_dir=True,
        evaluation_strategy = "epoch",
        learning_rate=3e-6,
        per_device_train_batch_size=batch_size_on_train,
        per_device_eval_batch_size=batch_size_on_eval,
        weight_decay=0.01,
        save_total_limit=200,
        num_train_epochs=200,
        predict_with_generate=True,
        fp16=False,
        save_strategy='epoch', # 要每个epoch都保存，要不保存不下来
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        gradient_accumulation_steps=1,
        run_name="english_mt5_learning_rate3e-6——batch_size8——length-512-128", # 现在只有GPU的信息，没有learning_rate这些类似的
        report_to='wandb', # 这个一定要，要不就没办法看到其他参数，只能看到系统的GPU那些
        logging_dir='./huggingface_logs',
        generation_max_length=128,
        generation_num_beams=10
        #run_name='test_rouge',
        # group_by_length=True,
        # warmup_steps=1000
        # metric_for_best_model="eval_loss", default会用eval_loss
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    patience = EarlyStoppingCallback(early_stopping_patience=4)
    # wandb_args = {"run_name":"learning_rate2e-5——batch_size48——length-1024-512"}
    # wandb = WandbCallback(wandb_args)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[patience], # wandb
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate(eval_dataset=tokenized_datasets['test'], metric_key_prefix='test', max_length=128, num_beams=10)