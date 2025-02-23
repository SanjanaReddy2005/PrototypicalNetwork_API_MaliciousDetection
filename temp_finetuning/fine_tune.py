import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

df = pd.read_csv('def_api.csv')

class WordMeaningDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data[idx]["class"]
        meaning = self.data[idx]["meaning"]

        # Tokenize word and meaning together
        inputs = self.tokenizer(word, meaning, 
                                padding="max_length", 
                                truncation=True, 
                                max_length=self.max_length, 
                                return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
        }

df = pd.read_csv('def_api.csv')
data = []
missed_data_points = []
for i,cla in enumerate(df['classes']):
  if isinstance(df['meainings'][i],str):
    data.append({'class':cla,'meaning':df['meainings'][i]})
  else:
    missed_data_points.append(cla)
print(data)

dataset = WordMeaningDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

import torch.optim as optim
from transformers import AdamW

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define loss function
loss_fn = torch.nn.MSELoss()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Define dummy target labels (modify this for your task)
        target = torch.zeros_like(logits)

        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

model.save_pretrained("bert_word_meaning_model")
tokenizer.save_pretrained("bert_word_meaning_model")

model.save_pretrained("bert_word_meaning_model")
tokenizer.save_pretrained("bert_word_meaning_model")
