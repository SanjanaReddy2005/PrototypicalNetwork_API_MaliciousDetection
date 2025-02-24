import torch
import torch.nn as nn
from transformers import BertModel

class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_filters=128, kernel_size=3):
        super(CNNEncoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # Global max pooling

    def forward(self, embeddings):
        """
        embeddings: Tensor of shape (batch_size, seq_length, embed_dim)
        """
        embeddings = embeddings.permute(0, 2, 1)  # Reshape for CNN (batch_size, embed_dim, seq_length)
        conv_output = self.conv1d(embeddings)  # Apply CNN
        conv_output = self.relu(conv_output)
        pooled_output = self.max_pool(conv_output).squeeze(-1)  # Apply max pooling
        
        return pooled_output  # (batch_size, num_filters)

# Example usage
batch_size = 2
seq_length = 64
embed_dim = 768
dummy_embeddings = torch.rand(batch_size, seq_length, embed_dim)  # Simulating BERT embeddings

encoder = CNNEncoder(embed_dim=embed_dim, num_filters=128, kernel_size=3)
sentence_vector = encoder(dummy_embeddings)

# print("Sentence representation shape:", sentence_vector.shape)  # Should be (batch_size, num_filters)
