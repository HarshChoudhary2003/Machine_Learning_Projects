import torch
import torch.nn as nn

class LSTMWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(LSTMWordPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len]
        embeds = self.embedding(x)  # shape: [batch_size, seq_len, embedding_dim]
        lstm_out, hidden = self.lstm(embeds, hidden)  # lstm_out shape: [batch_size, seq_len, hidden_dim]
        
        # We only predict the next word using the output of the *last* token in the sequence
        last_outputs = lstm_out[:, -1, :]  # shape: [batch_size, hidden_dim]
        last_outputs = self.dropout(last_outputs)
        
        logits = self.fc(last_outputs)  # shape: [batch_size, vocab_size]
        return logits, hidden
