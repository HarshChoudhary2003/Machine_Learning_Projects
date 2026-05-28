import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import LSTMWordPredictor

class SequenceDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def main():
    print("Loading preprocessed dataset...")
    with open("processed_dataset.json", "r") as f:
        data = json.load(f)
        
    inputs = data["inputs"]
    targets = data["targets"]
    vocab_size = data["vocab_size"]
    
    print(f"Loaded {len(inputs)} sequences with vocabulary size {vocab_size}.")
    
    # Dataset and DataLoader
    dataset = SequenceDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Model configuration
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.2
    epochs = 40
    learning_rate = 0.005
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = LSTMWordPredictor(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training loop...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_inputs.size(0)
            
        epoch_loss /= len(dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.4f}")
            
    # Save the PyTorch model
    model_path = "lstm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch model state dict to {model_path}")

if __name__ == "__main__":
    main()
