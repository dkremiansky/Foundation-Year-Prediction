import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)


from pathlib import Path
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import Trainer
from transformers import RobertaTokenizer

import random
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT_PATH = Path("results")


def manhattan_distance(a, b):
    return np.abs(a - b).sum()


def load_data():
    data_files = {
        'train': str('ready_train_100k.pkl'),
        'test': str('ready_test_100k.pkl')
    }
    raw_datasets = load_dataset("pandas", data_files=data_files)
    return raw_datasets


def set_tokens(model_name, raw_datasets, name_of_column):
    model_seq_classification = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model_seq_classification = model_seq_classification.to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    tokenized_datasets = raw_datasets.map(tokenizer, input_columns=name_of_column,
                                          fn_kwargs={"max_length": 256, "truncation": True, "padding": "max_length"})
    tokenized_datasets.set_format('torch')
    return model_seq_classification, tokenized_datasets


def classification_model(model, raw_datasets, tokenized_datasets):
    for split in tokenized_datasets:
        tokenized_datasets[split] = tokenized_datasets[split].add_column('label', raw_datasets[split]['founded'])

    def metric_fn(predictions):
        preds = predictions.predictions.argmax(axis=1)
        labels = predictions.label_ids
        return {'L1': manhattan_distance(np.array(labels, dtype=int), np.array(preds, dtype=int))}

    args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=8,
                     per_device_eval_batch_size=8, save_strategy='no',
                     greater_is_better=False, evaluation_strategy='epoch', do_train=True,
                     num_train_epochs=5, report_to='none', load_best_model_at_end=False, learning_rate=0.1)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=metric_fn
            )

    train_stat = trainer.train()


def transformers_model(name_of_column):
    model_name = 'roberta-base'
    raw_datasets = load_data()
    model, tokenized_datasets = set_tokens(model_name, raw_datasets, name_of_column)
    classification_model(model, raw_datasets, tokenized_datasets)
