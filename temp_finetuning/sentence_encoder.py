import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd

class SentenceEncoder(nn.Module):
    def __init__(self, bert_model_name="bert_word_meaning_model", embed_dim=768, num_filters=128, kernel_size=3):
        super(SentenceEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)  # BERT as embedding layer
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # Global max pooling

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Freeze BERT during encoding
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        embeddings = bert_outputs.last_hidden_state  # (batch_size, seq_length, embed_dim)
        embeddings = embeddings.permute(0, 2, 1)  # Convert to (batch_size, embed_dim, seq_length) for CNN
        
        conv_output = self.conv1d(embeddings)  # Apply CNN
        conv_output = self.relu(conv_output)
        pooled_output = self.max_pool(conv_output).squeeze(-1)  # Get fixed-size sentence vector
        
        return pooled_output  # (batch_size, num_filters)

# Example usage
tokenizer = BertTokenizer.from_pretrained("bert_word_meaning_model")

def encode_sentence(sentence, max_length=107):
    tokens = tokenizer(sentence, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return tokens["input_ids"], tokens["attention_mask"]

# Sample sentence with entity indicators
# sentence = "[E1] Barack Obama [/E1] was the 44th president of the [E2] United States [/E2]."

sentence = " "
input_ids, attention_mask = encode_sentence(sentence)

# Model forward pass
model = SentenceEncoder()
sentence_vector = model(input_ids, attention_mask)
print("Sentence representation shape:", sentence_vector.shape)  # Should be (1, num_filters)
