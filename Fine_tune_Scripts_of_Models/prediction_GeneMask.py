import argparse
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def load_data_from_file(file_path, tokenizer):
    try:
        df = pd.read_csv(file_path, delimiter='\t')
        texts = df['sequence'].tolist()
        labels = df['label'].tolist()
        inputs = tokenizer.batch_encode_plus(
            texts,
            padding='max_length',
            truncation=True,
            max_length=100,
            return_tensors="pt"
        )
        dataset = TensorDataset(
            inputs['input_ids'],
            inputs['attention_mask'],
            torch.tensor(labels)
        )
        return DataLoader(dataset, batch_size=32), labels
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def predict(model, file_path, tokenizer):
    model.eval()
    dataloader, true_labels = load_data_from_file(file_path, tokenizer)
    all_predictions = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]  # <-- Fix: use index instead of .logits
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
    logging.info("All predictions made successfully.")
    return all_predictions, true_labels



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the prediction script...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Call predict once
    predictions, true_labels = predict(model, args.file_path, tokenizer)

    # Save predictions to CSV
    output_file = "predictions.csv"
    with open(output_file, "w", newline="") as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(["index", "prediction"])
        for idx, pred in enumerate(predictions):
            writer.writerow([idx, pred])
    
    logging.info(f"Predictions completed and saved to {output_file}")

    # Compute metrics
    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    print("📊 Evaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Full classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))


if __name__ == "__main__":
    main()
