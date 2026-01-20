#import os
#os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, DefaultDataCollator, AutoConfig)

from sklearn.metrics import precision_recall_fscore_support,accuracy_score,roc_auc_score
import os
from datetime import datetime
import pandas as pd
# Define custom early stopping callback
from transformers import TrainerCallback, TrainerControl, EarlyStoppingCallback

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    


set_seed(42)



# Set CUDA memory management environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Empty CUDA cache
torch.cuda.empty_cache()

# Basic CUDA and device setup
print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current CUDA Device:", torch.cuda.current_device())
print("Device Name:", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model configuration
model_checkpoint = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_checkpoint, num_labels=2, ignore_mismatched_sizes=True)

# Check if custom token "U" needs to be added to tokenizer
if "U" not in tokenizer.get_vocab():
    tokenizer.add_tokens(["U"])
    print("✅ Added 'U' to tokenizer vocabulary")

# Initialize model with custom config
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config , trust_remote_code=True)
model = model.to(device)  # Move model to GPU

# Save updated tokenizer (optional)
tokenizer.save_pretrained("./modified_tokenizer")

# Set the path to your dataset
dataset_path = "/vol/space/caduceus/Data/GTF2F1"

# Load your dataset and tokenize it
dataset = load_dataset("csv", data_files={"train": f"{dataset_path}/train_sequences.csv", 
                                           "validation": f"{dataset_path}/validation_sequences.csv",
                                          "test": f"{dataset_path}/test_sequences.csv"})
print(dataset.column_names)  # Check what columns are available

class PrintOnEarlyStopCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if control.should_training_stop:
            print(f"🛑 Early stopping triggered at epoch {round(state.epoch, 2)} / step {state.global_step}")

    
# Tokenization function for dataset
def tokenize_function(examples):
    # Ensure that the tokenizer returns 'attention_mask'
    tokenized = tokenizer(
        examples['sequence'], 
        max_length=100, 
        truncation=True, 
        padding="max_length", 
        return_attention_mask=True  
    )
    return tokenized


# Apply tokenization

tokenized_data = dataset.map(tokenize_function, batched=True)
tokenized_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])


# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create data loaders
train_loader = DataLoader(tokenized_data["train"],  shuffle=True, collate_fn=data_collator)
for batch in train_loader:
    print(batch)
    break

val_loader = DataLoader(tokenized_data["validation"], collate_fn=data_collator)
test_loader = DataLoader(tokenized_data["test"], batch_size=8, collate_fn=data_collator)



# Define compute metrics function for evaluating the model
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(-1)

    precision, recall, fscore, _ = precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)

    try:
        auc = roc_auc_score(labels, probs[:, 1])  # Use class 1 probabilities
    except ValueError:
        auc = 0.0  # Fallback if AUC cannot be computed (e.g., only one class present)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': fscore,
        'auc': auc
    }

    
# Training arguments
# Training arguments with your specified configurations
training_args = TrainingArguments(
    output_dir='/vol/space/caduceus/Data/GTF2F1/output',  # Outputs and models will be saved here, dynamically set with date
    evaluation_strategy="steps",  # Evaluate every 'eval_steps'
    eval_steps=100,  # Number of steps to run evaluation
    save_steps=100,  # Number of steps to save the model
    save_total_limit=50,  # Max number of checkpoints to keep
    learning_rate=3e-5,
    save_strategy="steps",   
    warmup_steps=0,
    lr_scheduler_type="linear",  # Type of learning rate scheduler
    weight_decay=0.1,  # Weight decay rate
    gradient_accumulation_steps=1,  # Number of updates steps to accumulate before performing a backward/update pass
    num_train_epochs=30,  # Total number of training epochs
    per_device_train_batch_size=8,  # Batch size per device during training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    max_grad_norm=1,  # Maximum norm of the gradients
    logging_steps=1,  # Log metrics every 'logging_steps'
    report_to="none",  # Disable reporting to any online services
    fp16=False,  # Enable mixed precision training
    optim="adamw_torch",
    load_best_model_at_end=True ,
    metric_for_best_model="f1_score",
    greater_is_better=True,
    #deterministic=True,
    seed=42, 
    #early_stopping_patience=3, # Optimizer type: using PyTorch's native AdamW
) 



# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    compute_metrics=compute_metrics,
    callbacks = [
    EarlyStoppingCallback(early_stopping_patience=3),
    PrintOnEarlyStopCallback()
] 
)
# Run evaluation on the test set after training


# Train the model
trainer.train()


predictions = trainer.predict(tokenized_data["test"])

# Convert logits to class predictions
pred_labels = np.argmax(predictions.predictions, axis=-1)

# Extract the actual labels from the dataset
actual_labels = predictions.label_ids

# Create a DataFrame with actual and predicted labels
results_df = pd.DataFrame({
    "Actual Labels": actual_labels,
    "Predicted Labels": pred_labels
})

# Save the DataFrame to a CSV file
results_df.to_csv("test_predictions_caduceus_earlystop.csv", index=False)

test_loss = predictions.metrics.get("test_loss") if "test_loss" in predictions.metrics else (
    predictions.loss if hasattr(predictions, "loss") else None
)

# Compute metrics manually
precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, pred_labels, average='binary')
accuracy = accuracy_score(actual_labels, pred_labels)


# Print results
print("\n🔍 Test Set Evaluation Metrics:")
if isinstance(test_loss, (float, int)):
    print(f"   Loss:     {test_loss:.4f}")
else:
    print("   Loss:     N/A (not available from prediction output)")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Precision:{precision:.4f}")
print(f"   Recall:   {recall:.4f}")
print(f"   F1 Score: {f1:.4f}")

# Optionally, print out the path to confirm where the file is saved
print("Saved test predictions to 'test_predictions_earlystop.csv'")


# Save the fine-tuned model and tokenizer

tokenizer.save_pretrained('./fine_tuned_model')


