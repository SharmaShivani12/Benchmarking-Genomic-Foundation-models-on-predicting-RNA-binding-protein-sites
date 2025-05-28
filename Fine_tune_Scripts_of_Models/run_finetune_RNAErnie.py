"""
This module fine-tunes Ernie for RNA sequence classification using PyTorch and Hugging Face's Transformers library,
and saves test predictions to a CSV file along with the sequences and labels.
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from multimolecule import RnaTokenizer, RnaErnieForSequencePrediction,RnaErnieModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ========== Helper Functions
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ⚠️ VERY IMPORTANT


# ========== Argument Parsing
parser = argparse.ArgumentParser(description='Fine-tuning Ernie for RNA sequence classification.')
parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")

parser.add_argument('--model_name_or_path', type=str, required=True, help='Pretrained model name or path')
parser.add_argument('--train_file', type=str, required=True, help='Training data file')
parser.add_argument('--validation_file', type=str, required=True, help='Validation data file')
parser.add_argument('--test_file', type=str, required=True, help='Test data file')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model and results')
parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation')
parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
#parser.add_argument('--dataset_name', type=str, default='default_task', help='Name of the data for naming output files')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
...
args = parser.parse_args()
set_seed(args.seed)


num_train_epochs=args.num_train_epochs
# ========== Load Dataset
dataset = load_dataset('csv', data_files={
    'train': args.train_file,
    'validation': args.validation_file,
    'test': args.test_file
})
#print(dataset.column_names)

# ========== Preprocessing and Tokenization
tokenizer = RnaTokenizer.from_pretrained(args.model_name_or_path)

def tokenize_function(examples):
    #print("Original Data:", examples['sequence'])  # Debug print
    tokenized_inputs = tokenizer(examples['sequence'], padding="max_length", truncation=True, max_length=128)
    
    # Assume 'sequence' is not to be tensorized, just carry it along
    tokenized_inputs['original_sequence'] = examples['sequence']
    tokenized_inputs['labels'] = examples['label']  # Ensure 'labels' are numerical
    return tokenized_inputs

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

#print(dataset['train'].format)  # Check the format of the dataset



# ========== DataLoader Setup
train_loader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
validation_loader = DataLoader(dataset['validation'], batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=True)


# ========== Model Initialization
model = RnaErnieForSequencePrediction.from_pretrained(args.model_name_or_path, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

'''
def evaluate_thresholds(true_labels, probabilities, thresholds):
    for threshold in thresholds:
        # Apply the threshold to probability predictions
        predictions = (probabilities > threshold).astype('int32')
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        print(f"Threshold: {threshold:.2f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
'''



# ========== Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
total_steps = len(train_loader) * args.num_train_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

def train_or_evaluate(model, loader, optimizer, criterion, device, train=True, collect_results=False, return_logits=False):
    if train:
        model.train()
    else:
        model.eval()

    # Initialize metrics
     # Assuming 'classification' is the correct task type
    accuracy = torchmetrics.Accuracy(task='binary').to(device)
    precision = torchmetrics.Precision(num_classes=2, average='macro', task='binary').to(device)
    recall = torchmetrics.Recall(num_classes=2, average='macro', task='binary').to(device)
    f1_score = torchmetrics.F1Score(num_classes=2, average='macro',task='binary').to(device)
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_sequences = []
    all_logits = []

    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        labels = batch['labels'].to(device)
        if 'original_sequence' in batch:
            sequences = batch['original_sequence']
        else:
            sequences = ['Unavailable'] * len(labels) # Assuming sequence data is directly accessible

        with torch.set_grad_enabled(train):
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        accuracy.update(preds, labels)
        precision.update(preds, labels)
        recall.update(preds, labels)
        f1_score.update(preds, labels)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_sequences.extend(sequences)
        if return_logits:
            all_logits.extend(outputs.logits.detach().cpu().numpy()) 

    

    avg_loss = total_loss / len(loader)
    final_accuracy = accuracy.compute()
    final_precision = precision.compute()
    final_recall = recall.compute()
    final_f1 = f1_score.compute()

    # Reset metrics after each call to ensure fresh calculations each time
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()
    
    if collect_results:
      if return_logits:
        return avg_loss, final_accuracy, final_precision, final_recall, final_f1, all_preds, all_labels, all_sequences, all_logits
      else:
        return avg_loss, final_accuracy, final_precision, final_recall, final_f1, all_preds, all_labels, all_sequences
    else:
      if return_logits:
        return avg_loss, final_accuracy, final_precision, final_recall, final_f1, all_logits
      else:
        return avg_loss, final_accuracy, final_precision, final_recall, final_f1




# ========== Early Stopping Setup
early_stopping_patience = 3
best_val_f1 = 0.0
patience_counter = 0
best_val_loss = float('inf')

best_model_path = os.path.join(args.output_dir, 'best_model.pt')

# ========== Training Process with Early Stopping
for epoch in range(num_train_epochs):
    train_loss, train_acc, train_prec, train_rec, train_f1 = train_or_evaluate(
        model, train_loader, optimizer, criterion, device, train=True)
    scheduler.step()

    print(f"Epoch {epoch+1} - Training: Loss {train_loss:.4f}, Acc {train_acc:.4f}, "
          f"Precision {train_prec:.4f}, Recall {train_rec:.4f}, F1 {train_f1:.4f}")

    val_loss, val_acc, val_prec, val_rec, val_f1 = train_or_evaluate(
        model, validation_loader, optimizer, criterion, device, train=False)

    print(f"Epoch {epoch+1} - Validation: Loss {val_loss:.4f}, Acc {val_acc:.4f}, "
          f"Precision {val_prec:.4f}, Recall {val_rec:.4f}, F1 {val_f1:.4f}")

    # Early stopping logic
    if val_f1 > best_val_f1:
      best_val_f1 = val_f1
      patience_counter = 0
      torch.save(model.state_dict(), best_model_path)
      print(f"✅ New best model saved (F1: {val_f1:.4f}) to {best_model_path}")
    else:
      patience_counter += 1
      print(f"⚠️ F1 did not improve. Patience: {patience_counter}/{early_stopping_patience}")
      if patience_counter >= early_stopping_patience:
        print("🛑 Early stopping triggered based on F1.")
        break

# ========== Load Best Model Before Testing
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    print(f"✅ Loaded best model from {best_model_path} for testing.")

'''
for epoch in range(num_train_epochs):
    train_loss, train_acc, train_prec, train_rec, train_f1 = train_or_evaluate(
        model, train_loader, optimizer, criterion, device, train=True)
    scheduler.step()
    print(f"Epoch {epoch+1} - Training: Loss {train_loss:.4f}, Acc {train_acc:.4f}, Precision {train_prec:.4f}, Recall {train_rec:.4f}, F1 {train_f1:.4f}")
'''


val_loss, val_acc, val_prec, val_rec, val_f1 = train_or_evaluate(
    model, validation_loader, optimizer, criterion, device, train=False)
print(f" Validation: Loss {val_loss:.4f}, Acc {val_acc:.4f}, Precision {val_prec:.4f}, Recall {val_rec:.4f}, F1 {val_f1:.4f}")

# Testing after all epochs are complete
test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels, test_sequences, test_logits = train_or_evaluate(
    model, test_loader, optimizer, criterion, device, train=False, collect_results=True, return_logits=True)
print(f"Test: Loss {test_loss:.4f}, Acc {test_acc:.4f}, Precision {test_prec:.4f}, Recall {test_rec:.4f}, F1 {test_f1:.4f}")

'''
test_logits_array = np.array(test_logits)
probabilities = torch.sigmoid(torch.tensor(test_logits_array, dtype=torch.float32))
#probabilities_np = probabilities.detach().cpu().numpy() 
if isinstance(probabilities, torch.Tensor):
    probabilities_np = probabilities.detach().cpu().numpy()
else:
    probabilities_np = probabilities  # It's already a numpy array


thresholds = np.arange(0.1, 1.0, 0.1)

evaluate_thresholds(np.array(test_labels), probabilities_np, thresholds)
'''



# Create a DataFrame with sequences
test_results_df = pd.DataFrame({
    'Sequence': test_sequences,
    'Label': test_labels,
    'Prediction': test_preds
})

# Save to CSV
csv_path = os.path.join('test_results_RNAERNIE_with_earlystop_GTF2F1.csv')
test_results_df.to_csv(csv_path, index=False)
print(f"Saved test results to {csv_path}")

#model is predicting one class so need to change code.
'''
probabilities = torch.sigmoid(outputs.logits)  # Single probability for class '1'

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_thresholds(true_labels, probabilities, thresholds):
    for threshold in thresholds:
        # Apply the threshold to probability predictions
        predictions = (probabilities > threshold).int()
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        print(f"Threshold: {threshold:.2f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Example thresholds to test
thresholds = np.arange(0.1, 1.0, 0.1)

# Assuming 'true_labels' is a tensor or array containing the true binary labels
evaluate_thresholds(true_labels, probabilities.cpu().numpy(), thresholds)


'''
