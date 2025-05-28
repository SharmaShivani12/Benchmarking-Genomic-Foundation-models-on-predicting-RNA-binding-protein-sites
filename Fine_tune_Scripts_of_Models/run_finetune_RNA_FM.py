import os
import fm  # for development with RNA-FM
from pathlib import Path
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from Bio import SeqIO  # for file parsing
from sklearn.manifold import TSNE  # for dimension reduction
from sklearn.model_selection import train_test_split  # for splitting train/val/test
from tqdm.notebook import tqdm  # for showing progress
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

import random
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f'using {device} device')


# Load your CSV files
train_data = pd.read_csv("train_sequences.csv")  # Replace with your actual path
val_data = pd.read_csv("validation_sequences.csv")      # Replace with your actual path
test_data = pd.read_csv("test_sequences.csv")    # Replace with your actual path
# Check the structure of the data
print(train_data.head())  # Check the structure of the train data
print(val_data.head())
print(test_data.head())

# Assuming your CSV files have columns 'sequence' and 'label'
# If the column names are different, adjust the following code accordingly

# Encode sequences using simple character-level encoding
'''def encode_sequence(seq):
    char_to_int = {'A': 1, 'C': 2, 'G': 3, 'T': 4}  # Encoding for ACGT
    return [char_to_int[char] if char in char_to_int else 0 for char in seq]  # Handle unexpected chars

# Apply encoding
train_data['encoded_sequence'] = train_data['sequence'].apply(encode_sequence)
val_data['encoded_sequence'] = val_data['sequence'].apply(encode_sequence)
test_data['encoded_sequence'] = test_data['sequence'].apply(encode_sequence)'
'''
'''
# Pad sequences to the same length
max_len = 100
def pad_or_truncate_sequence(seq, max_len):
    # Truncate if the sequence is longer than max_len
    if len(seq) > max_len:
        return seq[:max_len]
    # Otherwise, pad the sequence to max_len
    else:
        return seq + [0] * (max_len - len(seq))

train_data['encoded_sequence'] = train_data['encoded_sequence'].apply(
    lambda x: x + [0] * (max_len - len(x))
)
val_data['encoded_sequence'] = val_data['encoded_sequence'].apply(
    lambda x: x + [0] * (max_len - len(x))
)
test_data['encoded_sequence'] = test_data['encoded_sequence'].apply(
    lambda x: x + [0] * (max_len - len(x))
)

def print_tokenized_data(df, n=5):
    """
    Prints the tokenized sequence for the first n samples in the dataframe.
    """
    for i in range(n):
        print(f"Sequence {i + 1}: {df['encoded_sequence'].iloc[i]}")

'''

# Print the first 5 tokenized sequences for training data


# Extract labels
labels = train_data['label'].values  # Assuming labels are stored in the 'label' column
num_class = len(np.unique(labels))

# Load RNA-FM model
fm_model, alphabet = fm.pretrained.rna_fm_t12(Path('/RNA-FM_pretrained.pth')  # Replace path if necessary
batch_converter = alphabet.get_batch_converter(
)

fm_model.to(device)  # use GPU if available
fm_model.eval()  # disables dropout for deterministic results
for param in fm_model.parameters():
    param.requires_grad = False 

'''
# RNA-FM embedding
chunk_size = 50
token_embeddings = np.zeros((len(labels), 1024, 640))
max_length = 100 
# divide all the sequences into chunks for processing due to GPU memory limit
for i in tqdm(range(0, len(train_data), chunk_size)):
    data = train_data.iloc[i:i + chunk_size]

    # Modify this line to include the labels and sequences together in the correct format
    batch_labels, batch_strs, batch_tokens = batch_converter(list(zip(data['label'], data['sequence'])))
    # Truncate or pad batch tokens
    batch_tokens = batch_tokens[:, :max_length]
   
    print("Batch sequences before batch_converter:")
    print(data['sequence'].tolist()[:5])  # Print first 5 sequences

    print("Batch tokens shape:", batch_tokens.shape)
    print("First tokenized sequence:", batch_tokens[0])

    # use GPU
    with torch.no_grad():
        results = fm_model(batch_tokens.to(device), repr_layers=[12])

    emb = results["representations"][12].cpu().numpy()

    token_embeddings[i:i + chunk_size, :emb.shape[1], :] = emb

print(token_embeddings.shape)
'''


def extract_rna_fm_embeddings(sequences, max_length=100):
    """
    Extracts RNA-FM embeddings while ensuring all sequences have the same length.
    
    - Truncates sequences longer than max_length
    - Pads sequences shorter than max_length
    """
    token_embeddings = []

    with torch.no_grad():  # No gradient computation needed
        for i in range(0, len(sequences), 50):  # Process in batches
            batch = sequences[i:i + 50]  # Chunking for memory efficiency
            
            # Tokenize the batch
            _, batch_strs, batch_tokens = batch_converter(list(zip(range(len(batch)), batch)))

            # Truncate or pad batch tokens to max_length
            batch_tokens = batch_tokens[:, :max_length]  # Truncate longer sequences
            pad_size = max_length - batch_tokens.shape[1]  # Compute padding size
            if pad_size > 0:
                batch_tokens = torch.nn.functional.pad(batch_tokens, (0, pad_size), value=0)  # Pad shorter sequences

            batch_tokens = batch_tokens.to(device)  # Move tokens to GPU
            results = fm_model(batch_tokens, repr_layers=[12])  # Get embeddings
            
            emb = results["representations"][12].cpu().numpy()  # Move to CPU
            token_embeddings.append(emb)

    return np.concatenate(token_embeddings, axis=0)  # Merge all batches

# Extract embeddings for training, validation, and test sets
train_embeddings = extract_rna_fm_embeddings(train_data['sequence'])
val_embeddings = extract_rna_fm_embeddings(val_data['sequence'])
test_embeddings = extract_rna_fm_embeddings(test_data['sequence'])

print("Embedding Shape:", train_embeddings.shape)  # Should be (num_samples, seq_len, 640)

# Dataset and DataLoader class
class RNATypeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # use the mean of the RNA-FM embedding along the sequence dimension
        # so that all the embeddings are converted from (L, 640) -> (640,)
        return np.mean(self.embeddings[idx], axis=0), self.labels[idx]
    
class RNATypeClassifier(nn.Module):
    def __init__(self, input_dim=640, num_classes=2, dropout_rate=0.6):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Fully connected layer
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.dropout(x)  # Apply dropout
        x = torch.mean(x, dim=1)  # Reduce sequence length
        x = x.contiguous()  # Ensure contiguous tensor
        return self.fc(x)  # Keep raw logits (No Softmax)


class RNABindingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Prepare DataLoaders
train_dataset = RNABindingDataset(train_embeddings, train_data['label'].values)
val_dataset = RNABindingDataset(val_embeddings, val_data['label'].values)
test_dataset = RNABindingDataset(test_embeddings, test_data['label'].values)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize classifier
model = RNATypeClassifier().to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)


train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []
max_val_acc = 0.0
best_epoch = 0
early_stopping_patience = 3
best_val_f1 = 0.0
patience_counter = 0
best_model_path = "rna_type_best_model.pt"


# Training loop
epochs = 70
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device).float(), y.to(device).long() 

        optimizer.zero_grad()
        output = model(x)
        #print(f"Output shape: {output.shape}, Target shape: {y.shape}")

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(output, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    
    ###  Compute Training Accuracy Using Collected Predictions
    train_preds = []
    train_targets = []

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device).float(), y.to(device).long()

            y_pred = model(x)
            preds = torch.argmax(y_pred, dim=1)

            train_preds.append(preds)
            train_targets.append(y)

    train_preds = torch.cat(train_preds, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    train_acc = (train_preds == train_targets).float().mean().cpu()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")


    # Validate the model
 # 🔹 Validate the model
    val_losses = []
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device).float(), y.to(device).long()

            y_pred = model(x)

            loss = criterion(y_pred, y)
            val_losses.append(loss.item())
            val_preds.append(torch.max(y_pred, 1)[1])
            val_targets.append(y)

    val_preds = torch.cat(val_preds, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    val_acc = (val_preds == val_targets).float().mean().cpu()

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    train_loss_history.append(total_loss / len(train_loader))  # Correct way to log train loss

    val_loss_history.append(np.mean(val_losses))

    print(f"Validation Loss: {np.mean(val_losses):.4f}, Validation Accuracy: {val_acc:.4f}")

    # Save the best model
    # Save the best model based on F1
    if f1_score(val_targets.cpu(), val_preds.cpu()) > best_val_f1:
      best_val_f1 = f1_score(val_targets.cpu(), val_preds.cpu())
      patience_counter = 0
      torch.save(model.state_dict(), best_model_path)
      print(f" Saved best model with F1: {best_val_f1:.4f}")
    else:
      patience_counter += 1
      print(f" F1 did not improve. Patience: {patience_counter}/{early_stopping_patience}")
      if patience_counter >= early_stopping_patience:
        print(" Early stopping triggered based on F1 score.")
        break

if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path} for testing.")
           
train_loss_history.append(total_loss / len(train_loader))  # Correct way to log train loss
val_loss_history.append(np.mean(val_losses))  # Keep this as is

# Load the best model before testing


# Test the model
test_preds = []
test_labels = []  # Store true labels

# Loop through test data
for batch in test_loader:
    x, y = batch
    x, y = x.to(device).float(), y.to(device).long()

    output = model(x)
    _, y_pred = torch.max(output.data, 1)  # Get predicted labels

    test_preds.append(y_pred.cpu().numpy())
    test_labels.append(y.cpu().numpy())  # Store true labels

# Convert to numpy arrays
test_preds = np.concatenate(test_preds)
test_labels = np.concatenate(test_labels)  # This replaces undefined y_test

# Compute Evaluation Metrics
correct = np.sum(test_preds == test_labels)
total = len(test_labels)
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
print(f'Total test samples: {total}, Correct: {correct}, Test Accuracy: {correct / total:.4f}')

# Create a DataFrame from test predictions and labels
test_results_df = pd.DataFrame({
    "True_Labels": test_labels,
    "Predictions": test_preds
})

# Save to CSV
test_results_df.to_csv("test_predictions_earlystop.csv", index=False)



