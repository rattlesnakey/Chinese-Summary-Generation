# -*- encoding:utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
import torch
torch.device('cuda')
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
import nltk
import numpy as np
import warnings
import nltk
nltk.download('punkt')
warnings.filterwarnings('ignore')

# DatasetDict({
#     train: Dataset({
#         features: ['document', 'summary', 'id'],
#         num_rows: 204045
#     })
#     validation: Dataset({
#         features: ['document', 'summary', 'id'],
#         num_rows: 11332
#     })
#     test: Dataset({
#         features: ['document', 'summary', 'id'],
#         num_rows: 11334
#     })
# })
# raw_datasets["train"][0]
# {'document': 'Recent reports have linked some France-based players with returns to Wales.\n"I\'ve always felt - and this is with my rugby hat on now; this is not region or WRU - I\'d rather spend that money on keeping players in Wales," said Davies.\nThe WRU provides £2m to the fund and £1.3m comes from the regions.\nFormer Wales and British and Irish Lions fly-half Davies became WRU chairman on Tuesday 21 October, succeeding deposed David Pickering following governing body elections.\nHe is now serving a notice period to leave his role as Newport Gwent Dragons chief executive after being voted on to the WRU board in September.\nDavies was among the leading figures among Dragons, Ospreys, Scarlets and Cardiff Blues officials who were embroiled in a protracted dispute with the WRU that ended in a £60m deal in August this year.\nIn the wake of that deal being done, Davies said the £3.3m should be spent on ensuring current Wales-based stars remain there.\nIn recent weeks, Racing Metro flanker Dan Lydiate was linked with returning to Wales.\nLikewise the Paris club\'s scrum-half Mike Phillips and centre Jamie Roberts were also touted for possible returns.\nWales coach Warren Gatland has said: "We haven\'t instigated contact with the players.\n"But we are aware that one or two of them are keen to return to Wales sooner rather than later."\nSpeaking to Scrum V on BBC Radio Wales, Davies re-iterated his stance, saying keeping players such as Scarlets full-back Liam Williams and Ospreys flanker Justin Tipuric in Wales should take precedence.\n"It\'s obviously a limited amount of money [available]. The union are contributing 60% of that contract and the regions are putting £1.3m in.\n"So it\'s a total pot of just over £3m and if you look at the sorts of salaries that the... guys... have been tempted to go overseas for [are] significant amounts of money.\n"So if we were to bring the players back, we\'d probably get five or six players.\n"And I\'ve always felt - and this is with my rugby hat on now; this is not region or WRU - I\'d rather spend that money on keeping players in Wales.\n"There are players coming out of contract, perhaps in the next year or so… you\'re looking at your Liam Williams\' of the world; Justin Tipuric for example - we need to keep these guys in Wales.\n"We actually want them there. They are the ones who are going to impress the young kids, for example.\n"They are the sort of heroes that our young kids want to emulate.\n"So I would start off [by saying] with the limited pot of money, we have to retain players in Wales.\n"Now, if that can be done and there\'s some spare monies available at the end, yes, let\'s look to bring players back.\n"But it\'s a cruel world, isn\'t it?\n"It\'s fine to take the buck and go, but great if you can get them back as well, provided there\'s enough money."\nBritish and Irish Lions centre Roberts has insisted he will see out his Racing Metro contract.\nHe and Phillips also earlier dismissed the idea of leaving Paris.\nRoberts also admitted being hurt by comments in French Newspaper L\'Equipe attributed to Racing Coach Laurent Labit questioning their effectiveness.\nCentre Roberts and flanker Lydiate joined Racing ahead of the 2013-14 season while scrum-half Phillips moved there in December 2013 after being dismissed for disciplinary reasons by former club Bayonne.',
#  'id': '29750031',
#  'summary': 'New Welsh Rugby Union chairman Gareth Davies believes a joint £3.3m WRU-regions fund should be used to retain home-based talent such as Liam Williams, not bring back exiled stars.'}



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
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
    dataset = load_dataset('json', data_files={'train': './data/train.json', 'validation':'./data/dev.json', 'test': 'test.json'} )
    metric = load_metric("rouge")
    # 这边改成mt5-small应该就可以了
    model_checkpoint = "google/mt5-small"
    prefix = "摘要:"
    max_input_length = 512
    max_target_length = 128
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 4
    args = Seq2SeqTrainingArguments(
        output_dir="sfzy-summarization",
        overwrite_output_dir=True,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=20,
        num_train_epochs=30,
        predict_with_generate=True,
        fp16=True,
        save_strategy='epoch', # 要每个epoch都保存，要不保存不下来
        dataloader_num_workers=4,
        metric_for_best_model="eval_rougeL",
        load_best_model_at_end=True
        # report_to=['tensorboard','wandb']
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    patience = EarlyStoppingCallback(early_stopping_patience=4)
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