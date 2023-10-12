# %%
from datasets import load_dataset

ds = load_dataset("conll2003")

# %%
text = """Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

from transformers import AutoTokenizer, BertPreTrainedModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokens = tokenizer(text)

tokens
# %%
from torch.utils.data import DataLoader

datasets = ds['train'].map(
    lambda e: tokenizer(e['tokens'], is_split_into_words=True), batched=True)

# %%
datasets.set_format(type='torch',
                    columns=['input_ids', 'token_type_ids', 'attention_mask'])

train_dataloader = DataLoader(datasets, 1)

# %%
next(iter(train_dataloader))

# %%

tokens = tokenizer(text).tokens()

# %%
tokenizer.convert_tokens_to_string(tokens)
# %%
token_ids = tokenizer(text)

# %%
tokenizer.decode(token_ids['input_ids'])

tags = ds["test"].features['ner_tags'].feature

# %%
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

model_name = "bert-base-cased"

bert_config = BertConfig.from_pretrained(model_name,
                                         num_labels=tags.num_classes)


class BertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # Load model body
        self.bert = BertModel(config, add_pooling_layer=False)
        # Set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Load and initialize weights
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                **kwargs):
        # Use model body to get encoder representations
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            **kwargs)
        # Apply classifier to encoder representation
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        # Calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # Return model output object
        return TokenClassifierOutput(loss=loss,
                                     logits=logits,
                                     hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertForTokenClassification.from_pretrained(
    model_name, config=bert_config).to(device)

# %%
input_ids = tokenizer.encode(text, return_tensors="pt")

# %%
bert_model(input_ids.to(device))

outputs = bert_model(input_ids.to(device)).logits

# %%
torch.argmax(outputs, dim=-1)

# %%
from torch.optim import AdamW

optimizer = AdamW(bert_model.parameters(), lr=5e-5)

bert_model.train()

for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = bert_model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    # lr_scheduler.step()
    optimizer.zero_grad()
