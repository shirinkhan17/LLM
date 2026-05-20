#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install transformers torch

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

gpt = AutoModel.from_pretrained(MODEL_NAME)
gpt.resize_token_embeddings(len(tokenizer))

hidden_size = gpt.config.hidden_size


# In[2]:


class GPTClassifier(nn.Module):
    def __init__(self, gpt, num_labels):
        super().__init__()
        self.gpt = gpt
        self.classifier = nn.Linear(gpt.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state

        # use final non-padding token representation
        last_index = attention_mask.sum(dim=1) - 1
        batch_index = torch.arange(hidden.size(0))

        final_hidden = hidden[batch_index, last_index]

        logits = self.classifier(final_hidden)

        return logits


# # Multiple-Choice Model

# In[8]:


class GPTMultipleChoice(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        self.scorer = nn.Linear(gpt.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        input_ids shape:
        (batch_size, num_choices, seq_len)
        """

        batch_size, num_choices, seq_len = input_ids.shape

        input_ids = input_ids.view(batch_size * num_choices, seq_len)
        attention_mask = attention_mask.view(batch_size * num_choices, seq_len)

        outputs = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state

        last_index = attention_mask.sum(dim=1) - 1
        batch_index = torch.arange(hidden.size(0))

        final_hidden = hidden[batch_index, last_index]

        scores = self.scorer(final_hidden)

        scores = scores.view(batch_size, num_choices)

        return scores


# # 1. Textual Entailment

# In[3]:


entailment_data = [
    {
        "premise": "A dog is running in the park.",
        "hypothesis": "An animal is outdoors.",
        "label": 0   # entailment
    },
    {
        "premise": "The man is sleeping.",
        "hypothesis": "The man is running.",
        "label": 1   # contradiction
    },
    {
        "premise": "A woman is reading a book.",
        "hypothesis": "The woman is a teacher.",
        "label": 2   # neutral
    }
]

entailment_labels = {
    0: "entailment",
    1: "contradiction",
    2: "neutral"
}

def format_entailment(example):
    return (
        "<start> "
        + example["premise"]
        + " <delim> "
        + example["hypothesis"]
        + " <extract>"
    )


# In[ ]:


model.eval()

text = format_entailment(entailment_data[0])

batch = tokenizer(
    [text],
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"]
    )

prediction = logits.argmax(dim=-1).item()

print(entailment_labels[prediction])


# # 2. Similarity
# 
#   For similarity, process both orders:

# In[4]:


similarity_data = [
    {
        "text1": "A man is playing guitar.",
        "text2": "A person is making music.",
        "label": 1
    },
    {
        "text1": "A cat is sleeping.",
        "text2": "A car is driving.",
        "label": 0
    }
]

def format_similarity(example):
    order1 = (
        "<start> "
        + example["text1"]
        + " <delim> "
        + example["text2"]
        + " <extract>"
    )

    order2 = (
        "<start> "
        + example["text2"]
        + " <delim> "
        + example["text1"]
        + " <extract>"
    )

    return order1, order2


# In[5]:


class GPTSimilarityClassifier(nn.Module):
    def __init__(self, gpt, num_labels=2):
        super().__init__()
        self.gpt = gpt
        self.classifier = nn.Linear(gpt.config.hidden_size, num_labels)

    def encode(self, input_ids, attention_mask):
        outputs = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state
        last_index = attention_mask.sum(dim=1) - 1
        batch_index = torch.arange(hidden.size(0))

        return hidden[batch_index, last_index]

    def forward(self, ids1, mask1, ids2, mask2):
        h1 = self.encode(ids1, mask1)
        h2 = self.encode(ids2, mask2)

        h = h1 + h2

        logits = self.classifier(h)

        return logits


# In[ ]:


sim_model = GPTSimilarityClassifier(gpt, num_labels=2)

optimizer = torch.optim.Adam(sim_model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    total_loss = 0

    for example in similarity_data:
        text1, text2 = format_similarity(example)

        batch1 = tokenizer(
            [text1],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        batch2 = tokenizer(
            [text2],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        label = torch.tensor([example["label"]])

        logits = sim_model(
            batch1["input_ids"],
            batch1["attention_mask"],
            batch2["input_ids"],
            batch2["attention_mask"]
        )

        loss = loss_fn(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# In[ ]:


sim_model.eval()

example = similarity_data[0]
text1, text2 = format_similarity(example)

batch1 = tokenizer([text1], padding=True, truncation=True, return_tensors="pt")
batch2 = tokenizer([text2], padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    logits = sim_model(
        batch1["input_ids"],
        batch1["attention_mask"],
        batch2["input_ids"],
        batch2["attention_mask"]
    )

prediction = logits.argmax(dim=-1).item()

print("Prediction:", "similar" if prediction == 1 else "not similar")
print("Actual:", "similar" if example["label"] == 1 else "not similar")


# # 3. Question Answering

# In[6]:


qa_data = [
    {
        "context": "The sky is clear and blue.",
        "question": "What color is the sky?",
        "answers": ["green", "blue", "red"],
        "label": 1
    }
]

def format_question_answering(example):
    choices = []

    for answer in example["answers"]:
        text = (
            "<start> "
            + example["context"]
            + " "
            + example["question"]
            + " <delim> "
            + answer
            + " <extract>"
        )

        choices.append(text)

    return choices


# In[ ]:


mc_model = GPTMultipleChoice(gpt)

optimizer = torch.optim.Adam(mc_model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    total_loss = 0

    for example in qa_data:
        choices = format_question_answering(example)

        batch = tokenizer(
            choices,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = batch["input_ids"].unsqueeze(0)
        attention_mask = batch["attention_mask"].unsqueeze(0)

        label = torch.tensor([example["label"]])

        scores = mc_model(input_ids, attention_mask)

        loss = loss_fn(scores, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# In[ ]:


mc_model.eval()

choices = format_question_answering(qa_data[0])

batch = tokenizer(
    choices,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

input_ids = batch["input_ids"].unsqueeze(0)
attention_mask = batch["attention_mask"].unsqueeze(0)

with torch.no_grad():
    scores = mc_model(input_ids, attention_mask)

predicted_choice = scores.argmax(dim=-1).item()

print("Predicted answer:", qa_data[0]["answers"][predicted_choice])
print("Correct answer:", qa_data[0]["answers"][qa_data[0]["label"]])


# # 4. Commonsense Reasoning

# In[7]:


commonsense_data = [
    {
        "context": "John put ice cream in the sun.",
        "question": "What happened next?",
        "answers": [
            "It melted.",
            "It became colder.",
            "It turned into a rock."
        ],
        "label": 0
    }
]

def format_commonsense(example):
    choices = []

    for answer in example["answers"]:
        text = (
            "<start> "
            + example["context"]
            + " "
            + example["question"]
            + " <delim> "
            + answer
            + " <extract>"
        )

        choices.append(text)

    return choices


# In[ ]:


mc_model = GPTMultipleChoice(gpt)

optimizer = torch.optim.Adam(mc_model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):

    total_loss = 0

    for example in commonsense_data:

        choices = format_commonsense(example)

        batch = tokenizer(
            choices,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = batch["input_ids"].unsqueeze(0)
        attention_mask = batch["attention_mask"].unsqueeze(0)

        label = torch.tensor([example["label"]])

        scores = mc_model(
            input_ids,
            attention_mask
        )

        loss = loss_fn(scores, label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# In[ ]:




