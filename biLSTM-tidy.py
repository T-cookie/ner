import torch
import torch.nn as nn
import torchtext

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

from datasets import load_dataset, load_metric

dataset = load_dataset("wnut_17", split='train')

tags = dataset.features['ner_tags'].feature
index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}


def tokenize_and_align_labels(example):
    tokenized_input = {
        word.lower(): example['ner_tags'][idx]
        for idx, word in enumerate(example['tokens']) if word.isalpha()
    }

    return tokenized_input


dataset = dataset.map(tokenize_and_align_labels, batched=False)
