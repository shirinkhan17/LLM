#!/usr/bin/env python
# coding: utf-8

# # Topic-Aware Transformer Embedding (Combined Explanation)
# 
# Normally Transformer input is:
# 
# ```text
# TokenEmbedding + PositionEmbedding
# ```
# 
# We extend this by adding:
# 
# ```text
# + TopicEmbedding
# ```
# 
# So the final Transformer input becomes:
# 
# ```text
# word meaning
# + word order
# + topic/domain information
# ```
# 
# This gives Multi-Head Attention richer context before attention computation begins.
# 
# ---
# 
# # Step 1 — Topic Words
# 
# Example topic words:
# 
# ```text
# ["doctor", "hospital", "medicine", "patient"]
# ```
# 
# These words represent the document topic:
# 
# ```text
# Medical / Health
# ```
# 
# ---
# 
# # Step 2 — Convert Topic Words to Embeddings
# 
# Each topic word is converted into a vector using the same embedding layer as tokens:
# 
# ```python
# topic_word_vec = self.token_embedding(topic_word_ids)
# ```
# 
# Example:
# 
# ```text
# doctor   → [0.2, 0.5, ...]
# hospital → [0.7, 0.1, ...]
# medicine → [0.4, 0.9, ...]
# patient  → [0.3, 0.2, ...]
# ```
# 
# Shape:
# 
# ```text
# (batch_size, num_topic_words, d_model)
# ```
# 
# Meaning:
# - each topic word has a dense semantic representation
# 
# ---
# 
# # Step 3 — Create One Topic Vector
# 
# All topic-word vectors are averaged:
# 
# ```python
# topic_vec = topic_word_vec.mean(dim=1)
# ```
# 
# Result:
# 
# ```text
# one vector representing the entire topic/domain
# ```
# 
# Example:
# 
# ```text
# Medical topic vector
# → [0.4, 0.5, 0.6, ...]
# ```
# 
# Shape:
# 
# ```text
# (batch_size, d_model)
# ```
# 
# This vector captures overall domain meaning.
# 
# ---
# 
# # Step 4 — Expand Topic Vector
# 
# The same topic vector is copied to every token position:
# 
# ```python
# topic_vec = topic_vec.unsqueeze(1).expand_as(token_vec)
# ```
# 
# Shape changes from:
# 
# ```text
# (batch_size, d_model)
# ```
# 
# to:
# 
# ```text
# (batch_size, seq_len, d_model)
# ```
# 
# Meaning:
# 
# | Token | Receives Topic Context |
# |---|---|
# | patient | medical topic |
# | diagnosed | medical topic |
# | medicine | medical topic |
# 
# Every token now carries document-level semantic context.
# 
# ---
# 
# # Step 5 — Position Embedding
# 
# Transformer attention alone does NOT know word order.
# 
# So positional embeddings are added.
# 
# Example positions:
# 
# ```text
# position 0 → [0.1, 0.3, ...]
# position 1 → [0.5, -0.2, ...]
# position 2 → [...]
# ```
# 
# These vectors are learned during training.
# 
# Purpose:
# - preserve sequence order
# - distinguish token locations
# 
# ---
# 
# # Step 6 — Merge All Embeddings
# 
# Final encoder input:
# 
# ```python
# x = token_vec + pos_vec + topic_vec
# ```
# 
# This merges:
# 
# | Embedding | Purpose |
# |---|---|
# | token_vec | word meaning |
# | pos_vec | word order |
# | topic_vec | domain/topic meaning |
# 
# ---
# 
# # Example
# 
# Sentence:
# 
# ```text
# "patient received medicine"
# ```
# 
# Topic words:
# 
# ```text
# ["doctor", "hospital", "medicine"]
# ```
# 
# Final representation becomes:
# 
# ```text
# patient
# +
# position information
# +
# medical topic context
# ```
# 
# So before attention even starts, the model already understands:
# 
# ```text
# this sentence is probably medical
# ```
# 
# ---
# 
# # Why This Helps
# 
# Some words are ambiguous.
# 
# Example:
# 
# ```text
# bank
# ```
# 
# Possible meanings:
# - financial institution
# - river side
# 
# Topic words:
# 
# ```text
# ["money", "loan", "finance"]
# ```
# 
# push the representation toward the financial meaning.
# 
# This helps attention heads focus on the correct semantic interpretation.
# 
# ---
# 
# # Overall Architecture
# 
# ```text
# Input Text
#     ↓
# Token Embedding
#     +
# Position Embedding
#     +
# Topic Word Embedding
#     ↓
# Topic-Aware Transformer Encoder
#     ↓
# Multi-Head Attention
#     ↓
# Context-Aware Representations
#     ↓
# Transformer Decoder
#     ↓
# Summary Generation
# ```
# 
# ---
# 
# # Big Picture
# 
# Topic-aware Transformers inject additional semantic/domain information directly into the encoder input:
# 
# ```text
# word meaning
# + sequence order
# + topic/domain meaning
# ```
# 
# This helps the model better understand:
# - document context
# - ambiguous words
# - domain-specific language
# - semantic focus
# 
# before Multi-Head Attention computes relationships between tokens.

# In[16]:


import torch
import torch.nn as nn
import math

torch.manual_seed(42)

# =====================================================
# 1. Tiny summarization dataset
# =====================================================
dataset = [
    {
        "text": "patient diagnosed diabetes received medicine hospital",
        "topic_words": ["patient", "doctor", "medicine", "hospital", "diabetes"],
        "summary": "patient received diabetes treatment"
    },
    {
        "text": "doctor prescribed medicine for sick patient",
        "topic_words": ["doctor", "patient", "medicine", "health", "treatment"],
        "summary": "doctor treated patient"
    },
    {
        "text": "bank approved loan for customer",
        "topic_words": ["bank", "money", "loan", "finance", "customer"],
        "summary": "bank approved loan"
    },
    {
        "text": "market prices affected money investment",
        "topic_words": ["market", "money", "investment", "finance", "prices"],
        "summary": "market affected investment"
    }
]


# In[17]:


# =====================================================
# 2. Vocabulary
# =====================================================

special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
all_words = set(special_tokens)

for item in dataset:
    all_words.update(item["text"].split())
    all_words.update(item["topic_words"])
    all_words.update(item["summary"].split())

vocab = {word: i for i, word in enumerate(sorted(all_words))}
id_to_word = {i: w for w, i in vocab.items()}

PAD = vocab["<PAD>"]
BOS = vocab["<BOS>"]
EOS = vocab["<EOS>"]
UNK = vocab["<UNK>"]


# In[23]:


# =====================================================
# 3. Encoding
# =====================================================
src_max_len = 8
topic_max_len = 5
summary_max_len = 7

def encode_text(text, max_len):
    #ids = [vocab[w] for w in text.split()]
    #Splits text into words and converts each word into its vocabulary ID.
    #If a word is not found, it uses UNK (unknown token ID).
    ids = [vocab.get(w, UNK) for w in text.split()]

    #Keeps only the first max_len tokens if sentence is too long.
    ids = ids[:max_len]
    #Adds PAD tokens to make all sequences the same length.
    ids += [PAD] * (max_len - len(ids))
    return ids

def encode_topic(words, max_len):
    ids = [vocab.get(w, UNK) for w in words]
    ids = ids[:max_len]
    ids += [PAD] * (max_len - len(ids))
    return ids

def encode_summary(summary, max_len):
    ids = [BOS] + [vocab.get(w, UNK) for w in summary.split()] + [EOS]
    ids = ids[:max_len]
    ids += [PAD] * (max_len - len(ids))
    return ids

src_ids = torch.tensor([encode_text(x["text"], src_max_len) for x in dataset])
topic_ids = torch.tensor([encode_topic(x["topic_words"], topic_max_len) for x in dataset])
summary_ids = torch.tensor([encode_summary(x["summary"], summary_max_len) for x in dataset])

decoder_input = summary_ids[:, :-1]
decoder_target = summary_ids[:, 1:]



# In[19]:


# =====================================================
# 4. Positional Embedding
# =====================================================
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        return self.position_embedding(positions)

# =====================================================
# 5. Normal Embedding: token + position
# =====================================================
class NormalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(max_len, d_model)

    def forward(self, token_ids):
        token_vec = self.token_embedding(token_ids)
        pos_vec = self.position_embedding(token_ids)
        return token_vec + pos_vec

# =====================================================
# 6. Topic-Aware Embedding: token + position + topic words
# =====================================================
class TopicAwareEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(max_len, d_model)

    def forward(self, token_ids, topic_word_ids):
        token_vec = self.token_embedding(token_ids)
        pos_vec = self.position_embedding(token_ids)

        topic_word_vec = self.token_embedding(topic_word_ids)

        # Average topic word embeddings
        topic_vec = topic_word_vec.mean(dim=1)

        # Add same topic vector to each input token
        topic_vec = topic_vec.unsqueeze(1).expand_as(token_vec)

        return token_vec + pos_vec + topic_vec

# =====================================================
# 7. Summarizer Model
# =====================================================
class Summarizer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        src_max_len=8,
        tgt_max_len=7,
        use_topic=False
    ):
        super().__init__()

        self.use_topic = use_topic

        if use_topic:
            self.encoder_embedding = TopicAwareEmbedding(
                vocab_size, d_model, src_max_len
            )
        else:
            self.encoder_embedding = NormalEmbedding(
                vocab_size, d_model, src_max_len
            )

        self.decoder_token_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_position_embedding = PositionalEmbedding(tgt_max_len, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=64,
            batch_first=True
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

    def make_causal_mask(self, size):
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def forward(self, src_ids, decoder_input_ids, topic_ids=None):
        if self.use_topic:
            encoder_x = self.encoder_embedding(src_ids, topic_ids)
        else:
            encoder_x = self.encoder_embedding(src_ids)

        decoder_x = (
            self.decoder_token_embedding(decoder_input_ids)
            + self.decoder_position_embedding(decoder_input_ids)
        )

        tgt_len = decoder_input_ids.size(1)
        causal_mask = self.make_causal_mask(tgt_len).to(decoder_input_ids.device)

        output = self.transformer(
            src=encoder_x,
            tgt=decoder_x,
            tgt_mask=causal_mask
        )

        logits = self.lm_head(output)

        return logits

# =====================================================
# 8. Train Function
# =====================================================
def train_model(model, epochs=300):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    for epoch in range(epochs):
        model.train()

        if model.use_topic:
            logits = model(src_ids, decoder_input, topic_ids)
        else:
            logits = model(src_ids, decoder_input)

        loss = loss_fn(
            logits.reshape(-1, len(vocab)),
            decoder_target.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# =====================================================
# 9. Generate Summary
# =====================================================
def generate_summary(model, text, topic_words=None, max_summary_len=7):
    model.eval()

    src = torch.tensor([encode_text(text, src_max_len)])

    if topic_words is not None:
        topic = torch.tensor([encode_topic(topic_words, topic_max_len)])
    else:
        topic = None

    generated = [BOS]

    for _ in range(max_summary_len - 1):
        decoder_in = torch.tensor([generated])

        with torch.no_grad():
            if model.use_topic:
                logits = model(src, decoder_in, topic)
            else:
                logits = model(src, decoder_in)

        next_token = logits[:, -1, :].argmax(dim=-1).item()

        if next_token == EOS:
            break

        generated.append(next_token)

    words = [
        id_to_word[i]
        for i in generated
        if i not in [BOS, EOS, PAD]
    ]

    return " ".join(words)

# =====================================================
# 10. Evaluation Function
# =====================================================
def evaluate_model(model, name):
    correct = 0
    total = len(dataset)

    print(f"\n================ {name} Evaluation ================\n")

    for item in dataset:
        if model.use_topic:
            pred_summary = generate_summary(
                model,
                item["text"],
                item["topic_words"]
            )
        else:
            pred_summary = generate_summary(
                model,
                item["text"],
                None
            )

        target_summary = item["summary"]

        print("Input:     ", item["text"])
        print("Target:    ", target_summary)
        print("Predicted: ", pred_summary)
        print()

        if pred_summary.strip() == target_summary.strip():
            correct += 1

    accuracy = correct / total

    print(f"{name} Exact Match Accuracy: {accuracy:.2f}")

    return accuracy



# In[20]:


# =====================================================
# 11. Train Normal Transformer
# =====================================================
print("\nTraining Normal Transformer WITHOUT topic words")

normal_model = Summarizer(
    vocab_size=len(vocab),
    use_topic=False,
    src_max_len=src_max_len,
    tgt_max_len=summary_max_len
)

train_model(normal_model, epochs=300)

normal_accuracy = evaluate_model(
    normal_model,
    "Normal Transformer"
)

# =====================================================
# 12. Train Topic-Aware Transformer
# =====================================================
print("\nTraining Topic-Aware Transformer WITH topic words")

topic_model = Summarizer(
    vocab_size=len(vocab),
    use_topic=True,
    src_max_len=src_max_len,
    tgt_max_len=summary_max_len
)

train_model(topic_model, epochs=300)

topic_accuracy = evaluate_model(
    topic_model,
    "Topic-Aware Transformer"
)

# =====================================================
# 13. Compare Accuracy
# =====================================================
print("\n================ Accuracy Comparison ================")
print(f"Without Topic Words Accuracy: {normal_accuracy:.2f}")
print(f"With Topic Words Accuracy:    {topic_accuracy:.2f}")


# In[22]:


# =====================================================
# 14. Compare Normal vs Topic-Aware on Unseen Text
# =====================================================

unseen_examples = [
    {
        "text": "sick patient received medicine doctor",
        "topic_words": ["patient", "doctor", "medicine", "health", "treatment"],
        "target": "doctor treated patient"
    },
    {
        "text": "customer received money loan bank",
        "topic_words": ["bank", "money", "loan", "finance", "customer"],
        "target": "bank approved loan"
    },
    {
        "text": "doctor diagnosed sick patient hospital",
        "topic_words": ["doctor", "patient", "hospital", "medicine", "health"],
        "target": "doctor treated patient"
    },
    {
        "text": "market investment affected customer money",
        "topic_words": ["market", "money", "investment", "finance", "prices"],
        "target": "market affected investment"
    }
]

normal_correct = 0
topic_correct = 0
total = len(unseen_examples)

print("\n================ Unseen Text Comparison ================\n")

for item in unseen_examples:
    text = item["text"]
    topic_words = item["topic_words"]
    target = item["target"]

    normal_summary = generate_summary(
        normal_model,
        text,
        topic_words=None
    )

    topic_summary = generate_summary(
        topic_model,
        text,
        topic_words=topic_words
    )

    if normal_summary.strip() == target.strip():
        normal_correct += 1

    if topic_summary.strip() == target.strip():
        topic_correct += 1

    print("Input Text:")
    print(text)

    print("\nTopic Words:")
    print(topic_words)

    print("\nTarget Summary:")
    print(target)

    print("\nNormal Transformer Summary:")
    print(normal_summary)

    print("\nTopic-Aware Transformer Summary:")
    print(topic_summary)

    print("\n----------------------------------------\n")

normal_unseen_acc = normal_correct / total
topic_unseen_acc = topic_correct / total

print("================ Unseen Accuracy Comparison ================")
print(f"Normal Transformer Unseen Accuracy:      {normal_unseen_acc:.2f}")
print(f"Topic-Aware Transformer Unseen Accuracy: {topic_unseen_acc:.2f}")


# In[ ]:





# In[ ]:




