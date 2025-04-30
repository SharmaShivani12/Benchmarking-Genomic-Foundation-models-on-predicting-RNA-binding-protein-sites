import argparse
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

def load_data_from_file(file_path, tokenizer):
    df = pd.read_csv(file_path, delimiter='\t')
    texts = df['sequence'].tolist()
    inputs = tokenizer.batch_encode_plus(
        texts,
        padding='max_length',  # You can also use True to pad to the maximum length in the batch
        truncation=True,
        max_length=100,  # Ensure this matches your desired sequence length
        return_tensors="pt"
    )
    dataset = TensorDataset(
        inputs['input_ids'], 
        inputs['attention_mask'], 
        torch.tensor(df['label'].values)  # Assuming you need labels for prediction
    )
    return DataLoader(dataset, batch_size=32)

def predict(model, file_path, tokenizer):
    model.eval()  # Set the model to evaluation mode
    dataloader = load_data_from_file(file_path, tokenizer)
    all_predictions = []
    device = next(model.parameters()).device 
    #all_predictions = []
    with torch.no_grad():  # Turn off gradients to speed up this part
        for batch in dataloader:
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())  # Collecting all predictions
    
    return all_predictions

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

    file_path = args.file_path
    df = pd.read_csv(file_path, delimiter='\t')
    true_labels = df['label'].values

    predictions = predict(model, file_path, tokenizer)

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='binary')
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info("\n" + classification_report(true_labels, predictions))

    # Save predictions
    output_file = "predictions.csv"
    with open(output_file, "w", newline="") as csvfile:
        import csv
        writer = csv.writer(csvfile)
        writer.writerow(["index", "prediction"])
        for idx, pred in enumerate(predictions):
            writer.writerow([idx, pred])

    logging.info("Predictions completed and saved to {}".format(output_file))




if __name__ == "__main__":
    main()
    
