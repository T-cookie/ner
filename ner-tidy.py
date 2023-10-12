import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer, BertPreTrainedModel
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments
from transformers import BertForTokenClassification
import torch

# load the data and tokenizer
model_name = "bert-base-uncased"
dataset = load_dataset("wikiann", name='en')

tags = dataset["train"].features['ner_tags'].feature
index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


def tokenize_and_align_labels(examples):
    tokenized_inputs = bert_tokenizer(examples["tokens"],
                                      truncation=True,
                                      is_split_into_words=True)
    labels = []
    for idx, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def encode_dataset(corpus):
    return corpus.map(
        tokenize_and_align_labels,
        batched=True,
    )


dataset = encode_dataset(dataset)

bert_config = AutoConfig.from_pretrained(model_name,
                                         num_labels=tags.num_classes)                                         
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertForTokenClassification.from_pretrained(
    model_name, config=bert_config).to(device)

num_epochs = 5
batch_size = 4
logging_steps = len(dataset["train"]) // batch_size
model_name = f"{model_name}-wiki-finetuned"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=num_epochs,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  evaluation_strategy="epoch",
                                  save_strategy='no',
                                  weight_decay=0.01,
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=False)

from datasets import load_metric

metric = load_metric("seqeval")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    y_pred = [[index2tag[p] for (p, l) in zip(prediction, label) if l != -100]
              for prediction, label in zip(predictions, labels)]
    y_true = [[index2tag[l] for (p, l) in zip(prediction, label) if l != -100]
              for prediction, label in zip(predictions, labels)]

    return metric.compute(predictions=y_pred, references=y_true)


from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(bert_tokenizer)

from transformers import Trainer

trainer = Trainer(bert_model,
                  args=training_args,
                  data_collator=data_collator,
                  compute_metrics=compute_metrics,
                  train_dataset=dataset["train"],
                  eval_dataset=dataset["validation"],
                  tokenizer=bert_tokenizer)

results = trainer.train()

trainer.save_state()

_, __, metrics = trainer.predict(dataset['test'])

trainer.save_metrics("predict", metrics)